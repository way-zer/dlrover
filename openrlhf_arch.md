## PPO Trainer

```mermaid
sequenceDiagram
    participant Trainer as PPOTrainer
    participant Rollout as RolloutWorker
    participant Reference as ReferenceWorker
    participant Actors as ActorWorker
    participant Reward as RewardWorker
    participant Critic as CriticWorker

    Trainer->>Rollout: init
    Rollout->>Rollout: vLLM
    Trainer->>Reference: init_model_from_pretrained
    Reference->>Reference: Model_Actor_eval
    Trainer->>Actors: init_model_from_pretrained
    Actors->>Actors: Model_Actor, tokenizer, Model_EMA, optimizer, scheduler
    Actors->>Actors: prepare_datasets
    Actors->>Actors: load_ckpt


    Trainer->>Reward: init_model_from_pretrained
    Reward->>Reward: Model_Reward_eval
    Trainer->>Critic: init_model_from_pretrained
    
    Trainer->>Actors: fit
    Actors->>Actors: Loop episode, sampling prompts, experience_maker.make_experience_list, advantage_estimator, ppo_train

    Trainer->>Actors: async_save_model
    Trainer->>Critic: async_save_model
```

## Note 4个月前进行了一次大改动，经验收集转移到Trainer中

控制流在Trainer内，核心流程：

- Loop episode
  - Loop sampling prompts
    - samples_generator.generate_samples
      - vllm.request
      - reward = remote_reward_model.get_rewards # required for dynamic_filtering
    - dynamic_filtering(samples)
    - experience_maker.make_experience_batch(samples)
      - reward = reward_model_group.forward
      - logprob = actor_model_group.forward
      - value = critic_model_group.forward
      - ref_logprob = reference_model_group.forward

    - actor.append(experience_batch) #batch
    - critic.append(experience_batch) #batch
    - ppo_train
      - critic.fit_async()
      - actor.fit()
      - actor._broadcast_to_vllm
      - wait critic.fit_async() done
  - save_logs_and_checkpoints
    - log % args.logging_steps
    - evaluate % args.eval_steps
    - samples_generator.generate_samples
    - vllm.request
    - reward = remote_reward_model.get_rewards
    - actor/critic.save_checkpoint % args.save_steps
