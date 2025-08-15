# PXDesignBench: A Unified Evaluation Suite for Protein Design

This repository provides a comprehensive suite of tools for protein design evaluation, integrating multiple state-of-the-art models with standardized pipelines. It supports both monomer and binder design, enabling thorough assessment across diverse aspects of protein design.


## 📂 Repository Structure

The codebase is organized into three main components:

- `metrics`: Scripts for evaluating multiple aspects of protein design, including sequence quality, structure quality, and designability.

- `tasks`: Pipelines for executing specific protein design evaluations (e.g., monomer, binder).

- `tools`: Wrappers for external models (e.g., Protenix, ProteinMPNN, AlphaFold2, ESMFold) to streamline integration.

### Supported Tasks & Tools
| **Task**   | **Sequence Generation** | **Structure Consistency**             |
|------------|-------------------------|---------------------------------------|
| **Monomer**| ProteinMPNN             | 🔹 ESMFold                            |
| **Binder** | ProteinMPNN             | 🔹 AlphaFold2 <br> 🔹 Protenix         |

---

## 📦 Dependencies

1. **Install Protenix** (version >= 0.5.3) – a high-accuracy structure prediction tool:  
```bash
pip install git+https://github.com/bytedance/Protenix.git@dev_design
```
If you encounter any issues during installation, please refer to the official [Protenix Installation Guide](https://github.com/bytedance/Protenix/tree/main?tab=readme-ov-file#-installation) for detailed instructions.

2. **Install additional dependencies** for ProteinMPNN, AlphaFold2, and ESMFold:
```bash
pip install git+https://github.com/sokrypton/ColabDesign.git --no-deps  # tested with colabdesign-1.1.3
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install transformers==4.51.3
```

3. **Download model weights:**
- **ESMFold:** [Download](https://huggingface.co/facebook/esmfold_v1/tree/main)  
- **AlphaFold2:** [Download](https://github.com/google-deepmind/alphafold?tab=readme-ov-file#model-parameters)  
- **ProteinMPNN:**
  - [CA model weights](https://github.com/dauparas/ProteinMPNN/tree/main/ca_model_weights)  
  - [Soluble model weights](https://github.com/dauparas/ProteinMPNN/tree/main/soluble_model_weights)  
  - [Vanilla model weights](https://github.com/dauparas/ProteinMPNN/tree/main/vanilla_model_weights)

The default weight paths are defined in **`eval_design/globals.py`**.  
After downloading, update the variables in `globals.py` to reflect your local directory structure.

### 📁 Model Weight Organization
Model weights for external tools are expected to be organized in a directory as follows:
```
├── af2
│   ├── LICENSE
│   ├── params_model_{1..5}.npz
│   ├── params_model_{1..5}_ptm.npz
│   ├── params_model_{1..5}_multimer_v3.npz
│
├── esmfold
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
│
├── mpnn
│   ├── ca_model_weights/...
│   ├── soluble_model_weights/...
│   └── vanilla_model_weights/...
```
**Note:** Required Protenix files (weights, CCD files, etc.) will be auto-downloaded on the first evaluation run.

---
## 🚀 Running the Evaluation
We provide demo scripts for both monomer and binder design evaluation.
**Monomer evaluation example:**
```bash
bash monomer_eval_demo.sh
```

**Binder evaluation example:**
```bash
bash binder_eval_demo.sh
```

### Input Formats

PXDesignBench supports multiple input modes, allowing you to evaluate protein designs flexibly.  
The basic CLI arguments are:
- ``--data_dir``: Directory containing input structures.
- ``--dump_dir``: Output directory for evaluation results.
- ``--is_mmcif``: Flag indicating whether input files are in **mmCIF** format (otherwise assumed **PDB**).

**JSON-based Input**
One can also provide a JSON configuration file describing the evaluation task. This format allows fine-grained control over task parameters and is particularly useful for batch evaluation.

Example JSON:
```python
{
    "task": ...,          # "monomer" or "binder"
    "pdb_dir": ...,       # directory containing the input PDB structures
    "name": ...,          # name of the task; used to locate "{pdb_dir}/{pdb_name}.pdb"
    "pdb_names": ...,     # list of PDB file names to evaluate 
    "cond_chains": ...,   # list of condition chains (only for binder evaluation)
    "binder_chains": ..., # list of binder chains (binder evaluation only; currently supports **one** binder chain)
    "out_dir": ...        # directory to store evaluation results
}
```
Key points:

- Binder tasks require `binder_chains` to be explicitly specified.
- Currently only one binder chain is supported; all other chains will be treated as condition chains.
- `pdb_names` defines the exact structures to evaluate. If omitted, all files in `pdb_dir` with valid suffixes will be evaluated.

**Directory-based Input**
Instead of JSON, one may provide a directory path directly to `--data_dir`.
In this case:

- If `file_name_list` is provided, only matching files will be evaluated.
- Otherwise, all files in the directory with valid extensions will be included.


### Binder Evaluation with Additional Metadata

Binder evaluation supports passing a JSON file to specify additional metadata beyond the default inputs.  
This is useful for advanced scenarios such as:

1. **Evaluating cropped sequences**  
   - If the sequence to be evaluated is a cropped segment of an original sequence, you can provide the **full original sequence** along with a `crop` field to specify the range used in evaluation.  
   - The crop range can be multiple ranges such as `"1-120,130-150"` (comma-separated ranges, 1-based indexing, inclusive).

2. **Providing precomputed MSA for Protenix filter**  
   - The Protenix filter requires the target chain's MSA.  
   - By default, the evaluation script will automatically call the Protenix MSA server to compute the MSA.  
   - If you have already computed the MSA locally, you can skip the server call by specifying the `msa` field with:
     - `precomputed_msa_dir`: Path to the local MSA directory.
     - `pairing_db`: `uniref100`.

**Example JSON input:**
```json
[
    {
        "proteinChain": {
            "sequence": "NAFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA",
            "label_asym_id": ["A0"],
            "use_msa": true,
            "msa": {
                "precomputed_msa_dir": "examples/msa/PDL1/0",
                "pairing_db": "uniref100"
            },
            "crop": "1-116"
        }
    }
]
```

### Multi-GPU / Distributed Evaluation
PXDesignBench exposes device IDs for each integrated model, enabling:
- Deployment across multiple GPUs for **parallel evaluation**.
- Integration into **DDP (Distributed Data Parallel)** training pipelines for **online evaluation tracking**.

For example, the following is a pseudocode snippet illustrating online evaluation tracking in the DDP model training pipeline:
```python
from eval_design.run import run_task
from protenix.utils.distributed import DIST_WRAPPER

@torch.no_grad()
def _evaluate_design(self):
    # Sampling stage
    self._inference_design()
    DIST_WRAPPER.barrier()

    # Load task list
    with open(self._get_eval_json_path(), "r") as f:
        all_eval_tasks = json.load(f)

    # Distribute tasks among workers
    task_indices = list(range(len(all_eval_tasks)))[DIST_WRAPPER.rank::DIST_WRAPPER.world_size]
    if not task_indices:
        self.local_print("No task to evaluate, skipping.")
        results = []
    else:
        self.local_print(f"Running {len(task_indices)} tasks...")
        results = [
            run_task(all_eval_tasks[i], self.configs.eval, device_id=DIST_WRAPPER.local_rank)
            for i in task_indices
        ]

    # Gather results
    all_eval_results = DIST_WRAPPER.all_gather_object(results)
    # Custom logging...
```

### Evaluation Process

- If `use_gt_seq=True`, the sequence from the input structure is used directly.
- If `use_gt_seq=False`, the tool will first run the assigned sequence generation model (e.g., ProteinMPNN) to generate sequences.
- Structure quality is then assessed using the corresponding structure prediction models:
  - Monomer → ESMFold
  - Binder → AlphaFold2 / Protenix
- Metrics include:
  - Self-consistency between predicted structures
  - Confidence scores from structure predictors (e.g., pLDDT, ipTM)
  - Shape-based metrics, e.g. secondary structure content (α-helix, β-sheet, loop ratios), radius of gyration, etc.
- Results are stored in a summary CSV for downstream analysis.

### Post-processing
PXDesignBench provides scripts for analyzing the **diversity** and **novelty** of generated protein structures.
To enable Foldseek-based diversity and novelty calculations, you must first install **Foldseek**, a structural alignment and similarity search tool.  
Foldseek is **not** bundled with PXDesignBench and must be installed separately.  
Please follow the official guide here: [Foldseek Installation](https://github.com/steineggerlab/foldseek#installation).

**Examples:**

- **Monomer post-processing**
```bash
python3 eval_design/scripts/postprocess_monomer.py --input_dir examples/monomer
```
- **Binder post-processing**
```bash
python3 eval_design/scripts/postprocess_binder.py --input_dir examples/binder --is_mmcif true
```

---

## 📚 Citing Related Work
If you use this repository, please cite the following works:

<details>
<summary>Protenix</summary>

```bibtex
@article{bytedance2025protenix,
  title={Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction},
  author={ByteDance AML AI4Science Team and Chen, Xinshi and Zhang, Yuxuan and Lu, Chan and Ma, Wenzhi and Guan, Jiaqi and Gong, Chengyue and Yang, Jincai and Zhang, Hanyu and Zhang, Ke and Wu, Shenghao and Zhou, Kuangqi and Yang, Yanping and Liu, Zhenyu and Wang, Lan and Shi, Bo and Shi, Shaochen and Xiao, Wenzhi},
  year={2025},
  journal={bioRxiv},
  publisher={Cold Spring Harbor Laboratory},
  doi={10.1101/2025.01.08.631967},
  URL={https://www.biorxiv.org/content/early/2025/01/11/2025.01.08.631967},
  elocation-id={2025.01.08.631967},
  eprint={https://www.biorxiv.org/content/early/2025/01/11/2025.01.08.631967.full.pdf},
}
```
</details>

<details>
<summary>ProteinMPNN</summary>

```bibtex
@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},
  pages={49--56},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
```
</details>

<details>
<summary>ESMFold</summary>

```bibtex
@article{lin2023evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and Verkuil, Robert and Kabeli, Ori and Shmueli, Yaniv and others},
  journal={Science},
  volume={379},
  number={6637},
  pages={1123--1130},
  year={2023},
  publisher={American Association for the Advancement of Science}
}
```
</details>

<details>
<summary>AlphaFold2</summary>

```bibtex
@article{jumper2021highly,
  title={Highly accurate protein structure prediction with AlphaFold},
  author={Jumper, John and Evans, Richard and Pritzel, Alexander and Green, Tim and Figurnov, Michael and Ronneberger, Olaf and Tunyasuvunakool, Kathryn and Bates, Russ and {\v{Z}}{\'\i}dek, Augustin and Potapenko, Anna and others},
  journal={nature},
  volume={596},
  number={7873},
  pages={583--589},
  year={2021},
  publisher={Nature Publishing Group UK London}
}
```
</details>

## Contributing 

We welcome contributions from the community to help improve the evaluation tool!

📄 Check out the [Contributing Guide](CONTRIBUTING.md) to get started.

✅ Code Quality: 
We use `pre-commit` hooks to ensure consistency and code quality. Please install them before making commits:

```bash
pip install pre-commit
pre-commit install
```

## Code of Conduct

We are committed to fostering a welcoming and inclusive environment.
Please review our [Code of Conduct](CODE_OF_CONDUCT.md) for guidelines on how to participate respectfully.


## Security

If you discover a potential security issue in this project, or think you may
have discovered a security issue, we ask that you notify Bytedance Security via our [security center](https://security.bytedance.com/src) or [vulnerability reporting email](sec@bytedance.com).

Please do **not** create a public GitHub issue.

## License

This project is licensed under the [Apache 2.0 License](./LICENSE). It is free for both academic research and commercial use.

