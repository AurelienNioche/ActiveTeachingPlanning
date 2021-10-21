from a2c.a2c import A2C

def main():
    def run_continuous_teaching(reward_type, gamma=1):
    forgets, repetitions = generate_learners_parameterization(
        n_users=N_USERS, n_items=N_ITEMS, seed=SEED_PARAM_LEARNERS)

    env = ContinuousTeaching(
        t_max=100,
        initial_forget_rates=forgets,
        initial_repetition_rates=repetitions,
        n_item=N_ITEMS,
        tau=0.9,
        delta_coeffs=np.array([3, 20]),
        penalty_coeff=0.2,
        reward_type=reward_type,
        gamma=gamma
    )

    m = A2C(env)

    # iterations = env.t_max * 5e5
    iterations = int(10e6)
    check_freq = env.t_max

    with ProgressBarCallback(env, check_freq) as callback:
        m.learn(iterations, callback=callback)

    plt.plot([np.mean(r) for r in callback.hist_rewards])
    plt.savefig('{}.png'.format(gamma))
    plt.show()
    return m

    teacher = A2C(env=env)


if __name__ == "__main__":
    main()
