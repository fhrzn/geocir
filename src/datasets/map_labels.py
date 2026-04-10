import argparse

import polars as pl
import timm.data

SYNSET_TO_CATEGORY = {
    # tower
    "n04460130": "tower",
    "n02814860": "tower",
    "n03047052": "tower",
    "n03519387": "tower",
    "n03688943": "tower",
    "n04556948": "tower",
    "n04562935": "tower",
    "n03347617": "tower",
    "n04206790": "tower",
    # castle_fortress
    "n02980441": "castle_fortress",
    "n03386011": "castle_fortress",
    "n04340935": "castle_fortress",
    "n02806088": "castle_fortress",
    "n03610098": "castle_fortress",
    "n02791665": "castle_fortress",
    "n02811936": "castle_fortress",
    "n02695627": "castle_fortress",
    "n03628511": "castle_fortress",
    "n02676938": "castle_fortress",
    "n03385557": "castle_fortress",
    "n04305323": "castle_fortress",
    "n03296328": "castle_fortress",
    "n03878066": "castle_fortress",
    "n03877845": "castle_fortress",
    "n03010915": "castle_fortress",
    "n03718458": "castle_fortress",
    # palace_manor
    "n03719053": "palace_manor",
    # church_cathedral
    "n02984203": "church_cathedral",
    "n02984061": "church_cathedral",
    "n03028079": "church_cathedral",
    "n02801184": "church_cathedral",
    "n03007130": "church_cathedral",
    "n02667576": "church_cathedral",
    "n02667379": "church_cathedral",
    "n02667478": "church_cathedral",
    "n03772077": "church_cathedral",
    "n03618982": "church_cathedral",
    "n04312432": "church_cathedral",
    # mosque
    "n03788195": "mosque",
    "n03767745": "mosque",
    "n03847471": "mosque",
    # temple_shrine
    "n04407435": "temple_shrine",
    "n04407686": "temple_shrine",
    "n04210390": "temple_shrine",
    "n04346328": "temple_shrine",
    "n04614655": "temple_shrine",
    "n03884778": "temple_shrine",
    "n03781244": "temple_shrine",
    "n03635032": "temple_shrine",
    "n04073948": "temple_shrine",
    "n04374735": "temple_shrine",
    # museum_gallery
    "n03412058": "museum_gallery",
    "n03661043": "museum_gallery",
    # theater_opera
    "n03849814": "theater_opera",
    "n04417809": "theater_opera",
    "n03801533": "theater_opera",
    "n02758134": "theater_opera",
    # stadium_arena
    "n04295881": "stadium_arena",
    "n03379204": "stadium_arena",
    "n03220692": "stadium_arena",
    "n03522003": "stadium_arena",
    "n02918112": "stadium_arena",
    "n02704949": "stadium_arena",
    "n03333610": "stadium_arena",
    # arch_gate
    "n04486054": "arch_gate",
    "n02733524": "arch_gate",
    "n03784896": "arch_gate",
    "n03448956": "arch_gate",
    "n04104384": "arch_gate",
    "n03975035": "arch_gate",
    "n04113765": "arch_gate",
    # monument_statue
    "n04306847": "monument_statue",
    "n03837869": "monument_statue",
    "n03074380": "monument_statue",
    "n03743902": "monument_statue",
    "n03810952": "monument_statue",
    "n02993194": "monument_statue",
    "n04313628": "monument_statue",
    "n04458633": "monument_statue",
    "n03743016": "monument_statue",
    # fountain
    "n03388043": "fountain",
    "n03241335": "fountain",
    # bridge
    "n02898711": "bridge",
    "n04366367": "bridge",
    "n04532670": "bridge",
    "n02953197": "bridge",
    "n04311004": "bridge",
    "n04492749": "bridge",
    "n04479939": "bridge",
    "n03122073": "bridge",
    "n03233744": "bridge",
    "n03379828": "bridge",
    "n04108822": "bridge",
    "n03865557": "bridge",
    # ruins
    "n04118635": "ruins",
    "n08492461": "ruins",
    "n02981024": "ruins",
    "n03727067": "ruins",
    "n02922292": "ruins",
    # garden_park
    "n03417345": "garden_park",
    "n03417749": "garden_park",
    "n04454908": "garden_park",
    # harbor_pier
    "n03721590": "harbor_pier",
    "n03216828": "harbor_pier",
    "n08633683": "harbor_pier",
    # modern_landmark
    "n04233124": "modern_landmark",
    "n04112654": "modern_landmark",
    "n03220513": "modern_landmark",
    "n03435593": "modern_landmark",
}


def build_im12k_mapping() -> dict[str, str]:
    timm_desc = timm.data.ImageNetInfo("imagenet-12k")
    return {
        timm_desc.label_name_to_description(k): v
        for k, v in SYNSET_TO_CATEGORY.items()
    }


def build_geotir_index(
    pred_path: str,
    latlon_path: str,
    img2lm_path: str,
    output_path: str,
) -> pl.DataFrame:
    im12k_mapping = build_im12k_mapping()

    df_pred = pl.read_csv(pred_path)
    df_pred_map = df_pred.filter(
        pl.col("pred_label").is_in(im12k_mapping.keys())
    ).with_columns(pl.col("pred_label").replace(im12k_mapping).alias("category"))

    df_w_latlon = pl.read_csv(latlon_path)
    df_landmark_ids = pl.read_csv(img2lm_path)

    result = (
        df_pred_map.join(df_landmark_ids, on="id", how="left")
        .join(df_w_latlon, on="landmark_id", how="inner")
        .select(
            [
                "id",
                "landmark_id",
                "latitude",
                "longitude",
                "country_code",
                "country",
                "region",
                "subregion",
                "category",
                pl.col("pred_label").alias("im21k_label"),
                "pred_score",
                pl.col("resolved_url").alias("wikimedia_url"),
                "geohack_url",
            ]
        )
    )

    result.write_csv(output_path)
    print(f"Wrote {len(result):,} rows to {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Build GeoTIR processed index CSV.")
    parser.add_argument("--pred", required=True, help="Path to index_predicted.csv")
    parser.add_argument("--latlon", required=True, help="Path to index_w_latlon_reversed.csv")
    parser.add_argument("--img2lm", required=True, help="Path to index_image_to_landmark.csv")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    build_geotir_index(
        pred_path=args.pred,
        latlon_path=args.latlon,
        img2lm_path=args.img2lm,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
