import asyncio
import base64
import os
import warnings
from typing import List, Literal

import polars as pl
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROMPT = """You are a strict JSON generator.

Input:
- IMAGE_DESCRIPTION (text)

Task:
Return JSON with keys: landmark_prob, reason, label.

Hard negatives (usually VERY LOW landmark_prob 0.00–0.24):
- vehicle is main subject (car, bus, train, truck)
- person/portrait is main subject
- food/indoor object is main subject
- accident/emergency/event is main subject
- generic street/building with no specific landmark focus

Label (choose ONE):
park, museum, church, mosque, temple, monastery, shrine, castle, palace, fort, bridge, 
tower, lighthouse, monument, cemetery, square, street, station, railway, airport, stadium, 
theatre, library, school, zoo, garden, mountain, waterfall, lake, river, canal, beach, 
island, cave, dam, archaeological site, building, settlement, landscape, port, unknown

landmark_prob rules:
- 0.85–0.99: primary subject is a specific landmark / famous place and is the focus
- 0.55–0.84: landmark is primary but unclear, far, or partly blocked
- 0.25–0.54: landmark is present but not the focus
- 0.00–0.24: no landmark focus or hard negative case

reason:
- 1 sentence only.
- Must state the primary subject and whether it is the focus.

Output rules:
- Output ONLY JSON. No extra text. No markdown.

JSON Schema:
{schema}
"""


async def main(args):
    # vllm server
    client = AsyncOpenAI(
        base_url=os.path.join(args.vllm_url, "v1"), api_key=args.vllm_token
    )

    # prepare prompt
    ImageLandmark = build_structured_output_cls()
    prompt = PROMPT.format(schema=ImageLandmark.model_json_schema())

    # load dataset
    df = pl.read_csv(args.data_path)
    img_paths = df["IMG_ID"].to_list()
    descriptions = df["reason"].to_list()

    # request
    all_responses = [None] * len(img_paths)

    tasks = []
    semaphore = asyncio.Semaphore(100)  # concurrent control
    for ix, (img_path, desc) in enumerate(zip(img_paths, descriptions)):
        msg = build_message(os.path.join(args.base_img_path, img_path), desc, prompt)
        tasks.append(
            asyncio.create_task(
                llm_call(
                    client=client,
                    semaphore=semaphore,
                    messages=msg,
                    response_format=ImageLandmark,
                    idx=ix,
                )
            )
        )

    # progress bar
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="llm call"):
        i, res = await coro
        all_responses[i] = res

    # postprocess dataset
    valid_item = []
    invalid_id = []
    for ii, res in enumerate(all_responses):
        if res:
            valid_item.append(res)
        else:
            invalid_id.append(ii)

    df_valid = pl.DataFrame(valid_item).with_row_index("idx")
    df_filtered = df.with_row_index("idx").filter(~pl.col("idx").is_in(invalid_id))
    df_filtered = df_filtered.drop("idx").with_row_index("idx")  # reindex
    df_filtered = df_filtered.join(df_valid, on="idx")
    df_filtered.write_csv(args.output_path)


async def llm_call(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    messages: List,
    response_format: BaseModel,
    idx: int,
):
    async with semaphore:
        # simple limiter
        # simple retry mechanism
        for it in range(3):
            await asyncio.sleep(0.5)
            # handle llm retry for invalid structured output
            try:
                response = await client.beta.chat.completions.parse(
                    # model="Qwen/Qwen3-VL-2B-Instruct",
                    model="OpenGVLab/InternVL3_5-14B-HF",
                    messages=messages,
                    response_format=response_format,
                    max_tokens=1024,
                    temperature=0,
                )
                return idx, response.choices[0].message.parsed.model_dump()
            except Exception:
                # retrying
                continue

        return idx, None


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def build_message(img_path: str, description: str, prompt: str):
    base64_image = encode_image(img_path)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {"type": "text", "text": description},
            ],
        }
    ]


def build_structured_output_cls():

    class ImageLandmark(BaseModel):
        landmark_prob: float = Field(
            description="probability score between 0 to 1 that indicates if the given image is classified as landmark"
        )
        reason: str = Field(
            description="descriptive reasoning of why such decision has been made."
        )
        label: Literal[
            "park",
            "museum",
            "church",
            "mosque",
            "temple",
            "monastery",
            "shrine",
            "castle",
            "palace",
            "fort",
            "bridge",
            "tower",
            "lighthouse",
            "monument",
            "cemetery",
            "square",
            "street",
            "station",
            "railway",
            "airport",
            "stadium",
            "theatre",
            "library",
            "school",
            "zoo",
            "garden",
            "mountain",
            "waterfall",
            "lake",
            "river",
            "canal",
            "beach",
            "island",
            "cave",
            "dam",
            "archaeological site",
            "building",
            "settlement",
            "landscape",
            "port",
            "unknown",
        ] = Field(description="best fit label to describe the given image")

    return ImageLandmark


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--vllm-url", default="http://localhost:3456")
    parser.add_argument("--vllm-token", default="token-is-ignored")
    parser.add_argument("--base-img-path", default="./datasets/mp16-reason")
    args = parser.parse_args()

    asyncio.run(main(args))
