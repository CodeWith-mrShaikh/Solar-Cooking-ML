"""
Kinetics Module for Solar Cooking ML Prediction System.
Implements the first-order thermal degradation and Arrhenius kinetics engine
for heat-sensitive vitamins (Vitamins C, B1, B2, B5, B6, B9).
Integrates dynamic transient food temperature profiles with trapezoidal integration:
    ln(C(t) / C0) = - ∫ k(T(τ)) dτ,  k(T) = A * exp(-Ea / (R * T))
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

# Thermodynamic and Universal Constants
R_GAS_CONSTANT = 8.314462618  # J / (mol · K)
CELSIUS_TO_KELVIN = 273.15

# Canonical 30-entry kinetics parameters compiled from published literature and Zenodo datasets
KINETICS_REGISTRY: List[Dict[str, Any]] = [
    # Vitamin C (Ascorbic Acid)
    {"vitamin": "Vitamin C", "food_source": "Citrus Fruits", "A_per_min": 1.25e6, "Ea_J_mol": 52300.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 40.0, "t_max_C": 95.0},
    {"vitamin": "Vitamin C", "food_source": "Leafy Greens", "A_per_min": 4.80e8, "Ea_J_mol": 68400.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 50.0, "t_max_C": 100.0},
    {"vitamin": "Vitamin C", "food_source": "Root Vegetables", "A_per_min": 3.10e5, "Ea_J_mol": 45200.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 45.0, "t_max_C": 95.0},
    {"vitamin": "Vitamin C", "food_source": "Fruit Juices", "A_per_min": 8.40e6, "Ea_J_mol": 58900.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 50.0, "t_max_C": 90.0},
    {"vitamin": "Vitamin C", "food_source": "Tubers", "A_per_min": 9.20e5, "Ea_J_mol": 49100.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 50.0, "t_max_C": 100.0},
    {"vitamin": "Vitamin C", "food_source": "McIlvaine buffer", "A_per_min": 65409.256945, "Ea_J_mol": 42771.622873, "origin": "Urvi sheet (fitted from k)", "doi": "10.5281/zenodo.13938958", "t_min_C": 70.0, "t_max_C": 105.0},
    {"vitamin": "Vitamin C", "food_source": "Rosehip nectar", "A_per_min": 2299592.0, "Ea_J_mol": 57386.182237, "origin": "Urvi sheet (fitted from k)", "doi": "10.1590/1678-457X.11417", "t_min_C": 70.0, "t_max_C": 95.0},
    {"vitamin": "Vitamin C", "food_source": "Strawberry nectar", "A_per_min": 22.95174, "Ea_J_mol": 22038.543184, "origin": "Urvi sheet (fitted from k)", "doi": "10.5281/zenodo.13938958", "t_min_C": 70.0, "t_max_C": 105.0},

    # Vitamin B1 (Thiamine)
    {"vitamin": "Vitamin B1", "food_source": "Legumes", "A_per_min": 2.10e11, "Ea_J_mol": 102500.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 60.0, "t_max_C": 120.0},
    {"vitamin": "Vitamin B1", "food_source": "Whole Grains", "A_per_min": 1.50e8, "Ea_J_mol": 75300.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 50.0, "t_max_C": 100.0},
    {"vitamin": "Vitamin B1", "food_source": "Meats", "A_per_min": 4.30e9, "Ea_J_mol": 85100.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 60.0, "t_max_C": 115.0},
    {"vitamin": "Vitamin B1", "food_source": "Root Vegetables", "A_per_min": 1.80e10, "Ea_J_mol": 91200.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 55.0, "t_max_C": 105.0},
    {"vitamin": "Vitamin B1", "food_source": "Dairy Products", "A_per_min": 6.20e8, "Ea_J_mol": 79800.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 60.0, "t_max_C": 100.0},

    # Vitamin B2 (Riboflavin)
    {"vitamin": "Vitamin B2", "food_source": "Dairy Products", "A_per_min": 2.10e4, "Ea_J_mol": 54200.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 60.0, "t_max_C": 110.0},
    {"vitamin": "Vitamin B2", "food_source": "Cereals", "A_per_min": 8.50e5, "Ea_J_mol": 61000.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 50.0, "t_max_C": 100.0},
    {"vitamin": "Vitamin B2", "food_source": "Leafy Greens", "A_per_min": 1.10e4, "Ea_J_mol": 48700.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 45.0, "t_max_C": 95.0},
    {"vitamin": "Vitamin B2", "food_source": "Legumes", "A_per_min": 4.90e5, "Ea_J_mol": 58300.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 50.0, "t_max_C": 100.0},

    # Vitamin B5 (Pantothenic Acid)
    {"vitamin": "Vitamin B5", "food_source": "Dairy Products", "A_per_min": 3.50e5, "Ea_J_mol": 67400.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 60.0, "t_max_C": 110.0},
    {"vitamin": "Vitamin B5", "food_source": "Meats", "A_per_min": 1.20e7, "Ea_J_mol": 78100.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 60.0, "t_max_C": 115.0},
    {"vitamin": "Vitamin B5", "food_source": "Whole Grains", "A_per_min": 1.80e5, "Ea_J_mol": 62800.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 50.0, "t_max_C": 100.0},
    {"vitamin": "Vitamin B5", "food_source": "Tubers", "A_per_min": 2.40e6, "Ea_J_mol": 71500.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 55.0, "t_max_C": 105.0},

    # Vitamin B6 (Pyridoxine)
    {"vitamin": "Vitamin B6", "food_source": "Meats", "A_per_min": 2.30e9, "Ea_J_mol": 82400.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 60.0, "t_max_C": 115.0},
    {"vitamin": "Vitamin B6", "food_source": "Whole Grains", "A_per_min": 5.00e6, "Ea_J_mol": 71200.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 50.0, "t_max_C": 100.0},
    {"vitamin": "Vitamin B6", "food_source": "Tubers", "A_per_min": 3.80e7, "Ea_J_mol": 76800.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 55.0, "t_max_C": 105.0},
    {"vitamin": "Vitamin B6", "food_source": "Legumes", "A_per_min": 8.10e8, "Ea_J_mol": 89500.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 55.0, "t_max_C": 110.0},
    {"vitamin": "Vitamin B6", "food_source": "Chickpea (Cicer arietinum L.)", "A_per_min": 2.734798, "Ea_J_mol": 21699.218427, "origin": "Urvi sheet (fitted from k)", "doi": "10.3390/foods13121847", "t_min_C": 25.0, "t_max_C": 85.0},

    # Vitamin B9 (Folic Acid)
    {"vitamin": "Vitamin B9", "food_source": "Leafy Greens", "A_per_min": 1.80e6, "Ea_J_mol": 64300.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 50.0, "t_max_C": 100.0},
    {"vitamin": "Vitamin B9", "food_source": "Legumes", "A_per_min": 6.40e8, "Ea_J_mol": 88700.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 55.0, "t_max_C": 110.0},
    {"vitamin": "Vitamin B9", "food_source": "Citrus Fruits", "A_per_min": 4.20e4, "Ea_J_mol": 56100.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 40.0, "t_max_C": 90.0},
    {"vitamin": "Vitamin B9", "food_source": "Whole Grains", "A_per_min": 9.10e6, "Ea_J_mol": 73900.0, "origin": "Om sheet (A, Ea given)", "doi": "Literature / Arrhenius benchmark", "t_min_C": 50.0, "t_max_C": 100.0}
]

# Descriptive metadata and real-world dietary examples for each kinetics food matrix
FOOD_SOURCE_METADATA: Dict[str, Dict[str, str]] = {
    "Tubers": {
        "label": "Tubers (Potato, Cassava, Yam - Root / Tuber Source)",
        "examples": "Potato (Root Tuber), Cassava, Yam, Sweet Potato",
        "description": "Starchy underground stem tubers and root sources rich in Vitamin C, B5, and B6."
    },
    "Root Vegetables": {
        "label": "Root Vegetables (Carrots, Beetroot, Radish - Root Source)",
        "examples": "Carrots, Beetroot, Radish, Turnips (Taproot Vegetables)",
        "description": "Underground taproots rich in dietary fiber, Vitamin C, and B1 (Thiamine)."
    },
    "Whole Grains": {
        "label": "Whole Grains (Rice, Wheat, Oats - Cereal Matrix)",
        "examples": "Rice, Wheat, Oats, Barley, Brown Rice",
        "description": "Intact grain kernels with germ and bran intact, major source of B-vitamins."
    },
    "Legumes": {
        "label": "Legumes (Beans, Lentils, Peas - Pulse Matrix)",
        "examples": "Chickpeas, Lentils (Dal), Black beans, Soybeans, Peas",
        "description": "High-protein pulse crops containing thiamine, riboflavin, B6, and folates."
    },
    "Chickpea (Cicer arietinum L.)": {
        "label": "Chickpea (Cicer arietinum L. / Chana - Legume)",
        "examples": "Garbanzo beans, Bengal gram, Kabuli chana",
        "description": "High-protein legume studied specifically in domestic cooking kinetics for Vitamin B6."
    },
    "Citrus Fruits": {
        "label": "Citrus Fruits (Orange, Lemon, Lime - Fruit Matrix)",
        "examples": "Orange, Lemon, Lime, Grapefruit, Amla",
        "description": "Acidic fruits containing high natural ascorbic acid (Vitamin C) and folates."
    },
    "Leafy Greens": {
        "label": "Leafy Greens (Spinach, Kale, Fenugreek - Green Vegetables)",
        "examples": "Spinach (Palak), Kale, Fenugreek (Methi), Cabbage",
        "description": "Tender green vegetables with high heat susceptibility for Vitamin C, B2, and B9."
    },
    "Fruit Juices": {
        "label": "Fruit Juices (Apple, Orange, Berry Juices - Liquid Matrix)",
        "examples": "Apple juice, Orange juice, Grape juice",
        "description": "Liquid fruit substrates where heat transfer is convective."
    },
    "Cereals": {
        "label": "Cereals (Corn, Millets, Sorghum - Grain Matrix)",
        "examples": "Corn (Maize), Sorghum (Jowar), Pearl millet (Bajra)",
        "description": "Staple cereal grains providing dietary riboflavin (B2)."
    },
    "Dairy Products": {
        "label": "Dairy Products (Milk, Yogurt - Animal Matrix)",
        "examples": "Cow milk, Buffalo milk, Yogurt, Whey",
        "description": "Liquid and semi-solid dairy containing thiamine, riboflavin, and pantothenic acid."
    },
    "Meats": {
        "label": "Meats (Poultry, Fish, Beef - Muscle Tissue)",
        "examples": "Chicken, Fish, Mutton, Beef",
        "description": "Animal protein tissue rich in B1, B5, and B6."
    },
    "McIlvaine buffer": {
        "label": "McIlvaine buffer (pH 3–7 Aqueous Model Buffer)",
        "examples": "Citrate-phosphate laboratory buffer",
        "description": "Controlled ionic strength aqueous model system for isolated ascorbic acid oxidation."
    },
    "Rosehip nectar": {
        "label": "Rosehip nectar (Rosa canina Extract - High Vit C)",
        "examples": "Rosehip juice/puree beverage",
        "description": "Ultra-rich natural ascorbic acid substrate tested across 70°C–95°C thermal pasteurization."
    },
    "Strawberry nectar": {
        "label": "Strawberry nectar (Fragaria ananassa - High Vit C)",
        "examples": "Strawberry beverage puree",
        "description": "Berry juice system with active anthocyanin pigments and high Vitamin C content."
    }
}

SOLAR_FOOD_METADATA: Dict[str, str] = {
    "Potato chips": "Potato chips (Tuber / Root Matrix - e.g. Potato)",
    "Rice": "Rice (Whole Grain / Cereal Matrix - e.g. Rice)",
    "Water": "Water (Aqueous Standard)",
    "Water (test medium)": "Water (Standard Solar Test Medium)",
    "Chamber air (proxy)": "Chamber air (Cavity Air Proxy)",
    "Water-temperature proxy from Tw": "Water-temperature proxy from Tw (Thermocouple Proxy)"
}

# Quick mapping dictionary for fast lookup: (vitamin, food_source) -> params
KINETICS_MAP: Dict[Tuple[str, str], Dict[str, Any]] = {
    (entry["vitamin"], entry["food_source"]): entry for entry in KINETICS_REGISTRY
}


def get_available_vitamins() -> List[str]:
    """Returns a sorted list of unique vitamins supported by the kinetics engine."""
    vitamins = sorted(list({entry["vitamin"] for entry in KINETICS_REGISTRY}))
    # Canonical order preference
    order_pref = ["Vitamin C", "Vitamin B1", "Vitamin B2", "Vitamin B5", "Vitamin B6", "Vitamin B9"]
    return [v for v in order_pref if v in vitamins] + [v for v in vitamins if v not in order_pref]


def get_all_food_sources() -> List[str]:
    """Returns all 14 unique food matrices / substrates in structured logical order."""
    return [
        "Tubers",
        "Root Vegetables",
        "Whole Grains",
        "Legumes",
        "Chickpea (Cicer arietinum L.)",
        "Citrus Fruits",
        "Leafy Greens",
        "Fruit Juices",
        "Cereals",
        "Dairy Products",
        "Meats",
        "McIlvaine buffer",
        "Rosehip nectar",
        "Strawberry nectar"
    ]


def get_vitamins_for_food_source(food_source: str) -> List[str]:
    """Returns all vitamins available for a given food source."""
    vits = [entry["vitamin"] for entry in KINETICS_REGISTRY if entry["food_source"] == food_source]
    order_pref = ["Vitamin C", "Vitamin B1", "Vitamin B2", "Vitamin B5", "Vitamin B6", "Vitamin B9"]
    res = [v for v in order_pref if v in vits] + [v for v in vits if v not in order_pref]
    return res if res else ["Vitamin C"]


def get_food_sources_for_vitamin(vitamin: str) -> List[str]:
    """Returns all food matrices/sources available for a given vitamin."""
    sources = [entry["food_source"] for entry in KINETICS_REGISTRY if entry["vitamin"] == vitamin]
    return sorted(sources)


def get_kinetics_entry(vitamin: str, food_source: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieves kinetic parameters (A, Ea) for a given vitamin and optional food source.
    If food_source is omitted or None, defaults to the primary food source.
    """
    available_sources = get_food_sources_for_vitamin(vitamin)
    if not available_sources:
        raise ValueError(f"Vitamin '{vitamin}' not found in kinetics database.")

    if food_source is None or food_source not in available_sources:
        food_source = available_sources[0]

    return KINETICS_MAP[(vitamin, food_source)]


def calculate_arrhenius_k(temperature_C: np.ndarray, A_per_min: float, Ea_J_mol: float) -> np.ndarray:
    """
    Computes Arrhenius reaction rate constant k(T) in min^-1:
        k(T) = A * exp(-Ea / (R * (T_C + 273.15)))
    """
    temp_C = np.asarray(temperature_C, dtype=float)
    temp_K = temp_C + CELSIUS_TO_KELVIN
    # Guard against zero or negative Kelvin
    temp_K = np.maximum(temp_K, 1e-6)
    k_values = A_per_min * np.exp(-Ea_J_mol / (R_GAS_CONSTANT * temp_K))
    return k_values


def simulate_vitamin_degradation(
    times_min: np.ndarray,
    temperatures_C: np.ndarray,
    vitamin: str,
    food_source: Optional[str] = None,
    solar_food_matrix: Optional[str] = None
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Simulates dynamic transient degradation of a vitamin under changing food temperatures
    over a sequence of time points using the trapezoidal integration method:
        ln(C_i / C_{i-1}) = - k_avg * Δt,  where k_avg = (k_{i-1} + k_i) / 2
        Retention_i = Retention_{i-1} * exp(-k_avg * Δt)
        Degradation_i = 100 - Retention_i

    Also evaluates the ideal isothermal benchmark where food is maintained at T_final.
    """
    times = np.asarray(times_min, dtype=float)
    temps = np.asarray(temperatures_C, dtype=float)

    if len(times) != len(temps):
        raise ValueError(f"Length mismatch: times ({len(times)}) vs temperatures ({len(temps)})")
    if len(times) < 2:
        raise ValueError("At least two time points are required for transient integration.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Time values must be strictly monotonically increasing.")

    entry = get_kinetics_entry(vitamin, food_source)
    food_source = entry["food_source"]
    A = entry["A_per_min"]
    Ea = entry["Ea_J_mol"]
    t_min_calib = entry.get("t_min_C", 40.0)
    t_max_calib = entry.get("t_max_C", 100.0)

    # 1. Real transient process: calculate k(T) at each minute
    k_real = calculate_arrhenius_k(temps, A, Ea)

    real_retention = np.zeros(len(times))
    real_retention[0] = 100.0

    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        k_avg = (k_real[i - 1] + k_real[i]) / 2.0
        real_retention[i] = real_retention[i - 1] * np.exp(-k_avg * dt)

    real_degradation = np.clip(100.0 - real_retention, 0.0, 100.0)
    real_retention = np.clip(real_retention, 0.0, 100.0)

    # 2. Ideal isothermal process: constant final predicted temperature
    t_final = temps[-1]
    k_ideal = float(calculate_arrhenius_k(np.array([t_final]), A, Ea)[0])
    ideal_retention = 100.0 * np.exp(-k_ideal * (times - times[0]))
    ideal_degradation = np.clip(100.0 - ideal_retention, 0.0, 100.0)
    ideal_retention = np.clip(ideal_retention, 0.0, 100.0)

    # 3. Rate of degradation (%/min)
    degradation_rate = np.gradient(real_degradation, times)
    peak_rate_idx = int(np.argmax(degradation_rate))
    peak_rate = float(degradation_rate[peak_rate_idx])
    peak_rate_time = float(times[peak_rate_idx])

    # 4. Check calibration temperature warnings
    outside_range_mask = (temps < t_min_calib) | (temps > t_max_calib)
    num_outside = int(np.sum(outside_range_mask))

    # 5. Food matrix matching check
    matrix_match = True
    if solar_food_matrix:
        solar_lower = str(solar_food_matrix).lower().strip()
        source_lower = str(food_source).lower().strip()
        matrix_match = (
            solar_lower in source_lower
            or source_lower in solar_lower
            or (solar_lower == "rice" and ("grain" in source_lower or "cereal" in source_lower))
            or (("potato" in solar_lower) and ("tuber" in source_lower or "root" in source_lower))
        )

    # Compile minute-by-minute timeline DataFrame
    timeline_df = pd.DataFrame({
        "time_min": times,
        "predicted_food_temp_C": np.round(temps, 2),
        "k_real_per_min": k_real,
        "real_retention_percent": np.round(real_retention, 3),
        "real_degradation_percent": np.round(real_degradation, 3),
        "ideal_retention_percent": np.round(ideal_retention, 3),
        "ideal_degradation_percent": np.round(ideal_degradation, 3),
        "temp_outside_calibration": outside_range_mask
    })

    # Summary dictionary
    summary = {
        "vitamin": vitamin,
        "food_source": food_source,
        "origin": entry["origin"],
        "doi": entry["doi"],
        "A_per_min": A,
        "Ea_J_mol": Ea,
        "Ea_kJ_mol": round(Ea / 1000.0, 2),
        "final_food_temp_C": round(float(t_final), 2),
        "max_food_temp_C": round(float(np.max(temps)), 2),
        "min_food_temp_C": round(float(np.min(temps)), 2),
        "avg_food_temp_C": round(float(np.mean(temps)), 2),
        "final_retention_percent": round(float(real_retention[-1]), 2),
        "final_degradation_percent": round(float(real_degradation[-1]), 2),
        "final_ideal_retention_percent": round(float(ideal_retention[-1]), 2),
        "final_ideal_degradation_percent": round(float(ideal_degradation[-1]), 2),
        "avg_retention_percent": round(float(np.mean(real_retention)), 2),
        "peak_degradation_rate_percent_per_min": round(peak_rate, 4),
        "peak_degradation_rate_time_min": peak_rate_time,
        "num_temps_outside_calibration": num_outside,
        "is_calibrated_range": (num_outside == 0),
        "calib_temp_min_C": t_min_calib,
        "calib_temp_max_C": t_max_calib,
        "solar_food_matrix": solar_food_matrix,
        "food_matrix_match": matrix_match
    }

    return summary, timeline_df


def get_all_kinetics_df() -> pd.DataFrame:
    """Returns the full 30-entry kinetics parameters registry as a formatted DataFrame."""
    df = pd.DataFrame(KINETICS_REGISTRY).copy()
    df["Ea_kJ_mol"] = (df["Ea_J_mol"] / 1000.0).round(2)
    df["example_foods"] = df["food_source"].map(lambda s: FOOD_SOURCE_METADATA.get(s, {}).get("examples", "Standard dietary food"))
    # Calculate k at reference temperature 80 °C (353.15 K) for easy comparative ranking
    df["k_at_80C_per_min"] = df.apply(
        lambda r: calculate_arrhenius_k(np.array([80.0]), r["A_per_min"], r["Ea_J_mol"])[0],
        axis=1
    )
    df["half_life_at_80C_min"] = (np.log(2) / np.maximum(df["k_at_80C_per_min"], 1e-12)).round(1)
    return df
