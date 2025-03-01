## MoLEx: Mixture of Layer Experts for Fine-Tuning with Sparse Upcycling
MoLEx: Mixture of Layer Experts for Fine-Tuning with Sparse Upcycling

https://openreview.net/forum?id=rWui9vLhOc&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions)

### Prerequisites

- pytorch
- The toolkit supports [Weights & Biases](https://docs.wandb.ai/) for monitoring jobs. If you use it, also install `wandb`.

### Usage


#### Set-up lora, NLU and NLG: 

- Run setup.py in the MoLEx folder 
- Run setup.py in the MoLEx/examples/NLU folder 
    Use the command:
    ```
    pip install -e .
    ```
- Install dependancies in MoLEx/examples/NLG/requirements.txt
- bash MoLEx/examples/NLG/download_pretrained_checkpoints.sh
- bash MoLEx/examples/NLG/eval/download_evalscript.sh
- Ensure that the transformers package source is from MoLEx/examples/NLU/src/transformers

#### Fine-tune RoBERTa-base on <glue_task>:
- possible <glue_task> include: MNLI, SST2, MRPC, CoLA, QNLI, QQP, RTE, STSB
    ```
    bash roberta_base_<glue_task>.sh
    ```

#### Fine-tune GPT2_M on E2E:
1. To train the GPT2_M model with MoLEx
    ```
    bash run_molex.sh
    ```
2. To generate outputs from the trained model using beam search
    ```
    bash run_beam.sh
    ```
3. Decode and evaluate outputs
    ```
    bash eval.sh
    ```


#### Wandb support:
- Add these flags to bash script with your project and job name
    ``` # Wandb: 
    --wandb_flag 
    --project-name test 
    --job-name test 
    ```


