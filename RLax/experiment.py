import haiku as hk
import jax


def run_loop(
    agent, environment, accumulator, seed,
    batch_size, train_episodes, evaluate_every, eval_episodes):
    """A simple run loop for examples of reinforcement learning with rlax

    Args:
        agent ([type]): [description]
        environment ([type]): [description]
        accumulator ([type]): [description]
        seed ([type]): [description]
        batch_size ([type]): [description]
        train_episodes ([type]): [description]
        evaluate_every ([type]): [description]
        eval_episodes ([type]): [description]
    """

    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    params = agent.initial_params(next(rng))
    learner_state = agent.initial_learner_state(params)

    print(f"Training agent for {train_episodes} episodes")
    for episode in range(train_episodes):

        timestep = environment.reset()
        accumulator.push(timestep, None)
        actor_state = agent.initial_actor_state()

        while not timestep.last():

            # acting
            actor_output, actor_state = agent.actor_step(
                params, timestep, actor_state, next(rng), evaluation=False
            )

            # agent-environment interaction
            timestep = environment.step(int(actor_output.actions))

            # accumulate experience
            accumulator.push(timestep, actor_output.actions)

            # Learning. 
            if accumulator.is_ready(batch_size):
                params, learner_state = agent.learner_step(
                    params, accumulator.sample(batch_size), learner_state, next(rng)
                )
        
        if not episode % evaluate_every:
            returns = 0. 

            for _ in range(eval_episodes):
                timestep = environment.reset()
                actor_state = agent.initial_actor_state()

                while not timestep.last():
                    actor_output, actor_state = agent.actor_step(
                        params, timestep, actor_state, next(rng), evaluation=True
                    )

                    timestep = environment.step(int(actor_output.actions))
                    returns += timestep.reward

                avg_returns = returns / eval_episodes
                print(f"Episode {episode:4d}: Average returns: {avg_returns:.2f} ")