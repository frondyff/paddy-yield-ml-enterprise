import pandas as pd

from paddy_yield_ml.pipelines import ablation_eval as ae


def test_extract_weather_water_features() -> None:
    dictionary_df = pd.DataFrame(
        [
            {"column_name": "A", "feature_group": "Weather"},
            {"column_name": "B", "feature_group": "Weather/water"},
            {"column_name": "C", "feature_group": "Farm profile"},
            {"column_name": "A", "feature_group": "Weather"},
        ]
    )
    out = ae.extract_weather_water_features(dictionary_df)
    assert out == ["A", "B"]


def test_build_ablation_scenarios() -> None:
    frame = pd.DataFrame(
        {
            "Variety": ["v1", "v2"],
            "Urea_40Days": [1, 2],
            "Rain_D1_D30": [10.0, 20.0],
            "Agriblock_feature": ["A", "B"],
            "Trash(in bundles)": [100, 120],
        }
    )
    scenarios = ae.build_ablation_scenarios(
        frame=frame,
        base_features=["Variety", "Trash(in bundles)"],
        weather_water_features=["Rain_D1_D30"],
        include_location_scenario=True,
    )
    assert scenarios["conservative_base"] == ["Variety"]
    assert scenarios["plus_weather_water"] == ["Variety", "Rain_D1_D30"]
    assert scenarios["plus_weather_water_and_location"] == ["Variety", "Rain_D1_D30", "Agriblock_feature"]
