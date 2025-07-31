import asyncio
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import art
from art.local import LocalBackend
from rcwa import Material, Layer, LayerStack, Source, Solver
import re
import ast

# Constants
start_wl = 0.32
stop_wl = 0.80
step_wl = 0.01
wavelengths = np.arange(start_wl, stop_wl + step_wl, step_wl)
DATA_DIR = Path("./data")  # Changed to your data directory

def load_test_dataset(data_dir=DATA_DIR):
    """Load the test dataset from JSON file"""
    test_path = data_dir / "test_dataset.json"
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_path}")
    
    with open(test_path, "r") as f:
        test_data = json.load(f)
    
    # Convert JSON data to TrainingExample objects
    examples = []
    for item in test_data:
        examples.append(
            TrainingExample(
                target_spectrum=item["target_spectrum"],
                true_order=item["true_order"],
                true_thickness=item["true_thickness"],
                prompt=item["prompt"],
                answer=item["answer"]
            )
        )
    print(f"\u2705 Loaded {len(examples)} test examples from {test_path}")
    return examples

# Classes for data structure
class TrainingExample:
    def __init__(self, target_spectrum, true_order, true_thickness, prompt, answer):
        self.target_spectrum = target_spectrum
        self.true_order = true_order
        self.true_thickness = true_thickness
        self.prompt = prompt
        self.answer = answer

def get_target_spectrum(order: list[str], thickness_nm: float) -> list[float]:
    """Simulate optical spectrum for given order and thickness"""
    source = Source(wavelength=start_wl)
    reflection_layer = Layer(n=1.0)
    transmission_layer = Layer(material=Material("Si"))
    thicknesses_microns = [thickness_nm * 0.001] * 4
    try:
        layers = [Layer(material=Material(m), thickness=t) for m, t in zip(order, thicknesses_microns)]
        stack = LayerStack(*layers, incident_layer=reflection_layer, transmission_layer=transmission_layer)
        solver = Solver(stack, source, (1, 1))
        result = solver.solve(wavelength=wavelengths)
        return np.array(result['TTot']).tolist()
    except Exception as e:
        print(f"Simulation failed: {e}")
        return []

def extract_predicted_order_and_thickness(response_text: str) -> tuple[list[str], int]:
    """Extract predicted order and thickness from model response"""
    try:
        # Match "Order: [...]" and "Thickness: [number]nm"
        match = re.search(r"Order:\s*(\[[^\]]+\])\s*,\s*Thickness:\s*(\d+)\s*nm", response_text)
        if match:
            order_str, thickness_str = match.groups()
            order = ast.literal_eval(order_str)  # Safely evaluate list string
            thickness = int(thickness_str)
            return order, thickness
    except Exception as e:
        print(f"Failed to extract prediction: {e}")
    return [], -1

async def main():
    # Load test data instead of trajectories
    test_examples = load_test_dataset()

    # Load model
    model = art.TrainableModel(
        name="optical-rl-agent",
        project="optical-rl-training",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        trainable=False,
    )
    model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(max_seq_length=16000),
        engine_args=art.dev.EngineArgs(
            enforce_eager=True,
            gpu_memory_utilization=0.85,
            num_scheduler_steps=1,
        ),
    )
    await model.register(LocalBackend(in_process=True, path="./.art"))

    results = []
    correct_order, correct_thickness = 0, 0

    for i, example in enumerate(test_examples):
        true_order = example.true_order
        true_thickness = example.true_thickness
        target_spectrum = example.target_spectrum

        prompt = example.prompt

        response = await model.openai_client().chat.completions.create(
            model="optical-rl-agent",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        output = response.choices[0].message.content
        print(f"[Output]: {output}")
        pred_order, pred_thickness = extract_predicted_order_and_thickness(output)
        print(f"Predicted order: {pred_order}, Predicted Thickness: {pred_thickness}")

        pred_spectrum = get_target_spectrum(pred_order, pred_thickness)
        cos_sim = float(cosine_similarity([target_spectrum], [pred_spectrum])[0][0]) if pred_spectrum else 0.0
        mse = float(mean_squared_error(target_spectrum, pred_spectrum)) if pred_spectrum else -1

        correct_order += int(pred_order == true_order)
        correct_thickness += int(pred_thickness == true_thickness)

        results.append({
            "index": i,
            "true_order": true_order,
            "true_thickness": true_thickness,
            "pred_order": pred_order,
            "pred_thickness": pred_thickness,
            "cosine_similarity": cos_sim,
            "mse": mse,
            "response": output,
        })

        print(f"[{i+1}/{len(test_examples)}] \u2713")

    summary = {
        "model_name": model.name,
        "timestamp": datetime.now().isoformat(),
        "accuracy": {
            "order": f"{correct_order} / {len(test_examples)} = {correct_order / len(test_examples) * 100:.2f}%",
            "thickness": f"{correct_thickness} / {len(test_examples)} = {correct_thickness / len(test_examples) * 100:.2f}%"
        },
        "results": results
    }

    out_path = Path("evaluations") / f"evaluation_{model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nEvaluation saved to: {out_path}")

if __name__ == "__main__":
    asyncio.run(main())