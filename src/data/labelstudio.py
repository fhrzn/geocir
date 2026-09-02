"""
Rewrite the host portion of task image URLs in-place, without resyncing
the storage source (so existing annotations are preserved).

Use case: tasks were created via Cloud Storage sync while LABEL_STUDIO_HOST
was set to an ngrok domain, so each task's data.image field has that ngrok
URL baked in absolutely. Switching the env var afterward doesn't retroactively
fix already-stored task data - this script does that directly via the API.

Usage:
    python rewrite_task_host.py \
        --base-url http://localhost:7766 \
        --api-token YOUR_TOKEN \
        --project-id 13 \
        --old-host https://awake-visually-catfish.ngrok-free.app \
        --new-host http://localhost:7766

Run this while accessing Label Studio via whichever host you're
currently authenticated against (matches --base-url).
"""

import argparse
import sys

import requests
from tqdm import  tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="Current Label Studio base URL you're authenticated against")
    parser.add_argument("--api-token", required=True)
    parser.add_argument("--project-id", required=True, type=int)
    parser.add_argument("--old-host", required=True, help="Host prefix currently baked into task image URLs, e.g. https://your-domain.ngrok-free.app")
    parser.add_argument("--new-host", required=True, help="Host prefix to replace it with, e.g. http://localhost:7766")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    old_host = args.old_host.rstrip("/")
    new_host = args.new_host.rstrip("/")
    headers = {"Authorization": f"Token {args.api_token}"}

    resp = requests.get(
        f"{base_url}/api/tasks",
        headers=headers,
        params={"project": args.project_id, "page_size": 10000},
    )
    resp.raise_for_status()
    payload = resp.json()
    tasks = payload.get("tasks", payload if isinstance(payload, list) else [])

    if not tasks:
        print(f"No tasks found for project {args.project_id}.")
        return

    n_changed = 0
    for task in tqdm(tasks, desc="updating base url..."):
        task_id = task["id"]
        image_url = task.get("data", {}).get("image", "")
        if old_host not in image_url:
            continue

        new_url = image_url.replace(old_host, new_host)
        n_changed += 1

        if args.dry_run:
            print(f"[dry-run] task {task_id}: {image_url} -> {new_url}")
            continue

        new_data = dict(task["data"])
        new_data["image"] = new_url
        r = requests.patch(
            f"{base_url}/api/tasks/{task_id}",
            headers=headers,
            json={"data": new_data},
        )
        if r.status_code not in (200, 201):
            print(f"  [warn] failed to update task {task_id}: {r.status_code} {r.text}", file=sys.stderr)

    print(f"\n{'Would update' if args.dry_run else 'Updated'} {n_changed}/{len(tasks)} tasks.")
    print("Annotations are untouched - this only rewrites the data.image field.")


if __name__ == "__main__":
    main()