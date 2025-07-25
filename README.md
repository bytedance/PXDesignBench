# Protenix Protein Design Evaluation Tools

This repository provides a comprehensive suite of tools for evaluating protein design tasks. It integrates multiple state-of-the-art models and offers scripts to assess various aspects of protein design. 

The codebase is organized into three main components:

- `metrics`: Scripts to evaluate different aspects of protein design, such as sequence quality, structure quality, and designability.

- `tasks`: Pipelines for running specific protein design tasks, including monomer and binder design.

- `tools`: Wrappers for external models such as Protenix, ProteinMPNN, AlphaFold2 and ESMFold to streamline evaluations.

Current supported tasks and tools are:
| **Task**   | **Sequence Generation** | **Structure Consistency**             |
|------------|-------------------------|---------------------------------------|
| **Monomer**| ProteinMPNN             | 🔹 ESMFold                            |
| **Binder** | ProteinMPNN             | 🔹 AlphaFold2 <br> 🔹 Protenix         |


## Dependencies

1. **Install Protenix** – a high-accuracy structure prediction tool:  
   [Protenix Installation Guide](https://github.com/bytedance/Protenix/tree/main?tab=readme-ov-file#-installation)

2. **Install additional dependencies** for ProteinMPNN, AlphaFold2, and ESMFold:
```bash
pip install git+https://github.com/sokrypton/ColabDesign.git --no-deps  # tested with colabdesign-1.1.3
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install transformers==4.51.3
```

3. **Download model weights**
- **ESMFold:** [Download](https://huggingface.co/facebook/esmfold_v1/tree/main)  
- **AlphaFold2:** [Download](https://github.com/google-deepmind/alphafold?tab=readme-ov-file#model-parameters)  
- **ProteinMPNN:**  
  - [CA model weights](https://github.com/dauparas/ProteinMPNN/tree/main/ca_model_weights)  
  - [Soluble model weights](https://github.com/dauparas/ProteinMPNN/tree/main/soluble_model_weights)  
  - [Vanilla model weights](https://github.com/dauparas/ProteinMPNN/tree/main/vanilla_model_weights)

The default paths for model weights are defined in **`eval_design/globals.py`**.  
After downloading the required weights for ESMFold, AlphaFold2, and ProteinMPNN, please update the corresponding variables in `globals.py` to match your local directory structure.


## Running the Evaluation
### Monomer Design

Run a monomer design evaluation demo:

```bash
bash monomer_eval_demo.sh
```

### Binder Design

Run a binder design evaluation demo:

```bash
bash binder_eval_demo.sh
```


### 📚 Citing Related Work
If you use this repository, please cite the following works:
```
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

