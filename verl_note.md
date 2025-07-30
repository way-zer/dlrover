## 主要流程

- init
  - tokenizer, processor
  - dataloader(train/validation)
- fit
  - load_ckpt
    - RG_ACTOR.load_ckpt
    - RG_CRITIC.load_ckpt
    - dataloader.load_state
  - _validate #val_before_train
    - Loop sampling val_dataloader
      - RG_ACTOR.generate_sequences
  - Loop epoch
    - Loop sampling prompts
      - RG_ACTOR.generate_sequences
      - reward = RG_REWARD.compute_rm_score
      - old_log_prob = RG_ACTOR.compute_log_prob
      - ref_log_prob = RG_REF.compute_log_prob
      - values = RG_CRITIC.compute_value
      - compute_advantage
      - RG_CRITIC.update_critic
      - RG_ACTOR.update_actor
      - _validate % test_freq
      - _save_ckpt % save_freq
        - RG_ACTOR.save_ckpt
        - RG_CRITIC.save_ckpt
        - dataloader.save_state

RPC均为阻塞调用，而且直接基于TensorDict拷贝传值，没有异步优化...
ACTOR推理->ACTOR
