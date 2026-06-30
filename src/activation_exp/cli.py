def train():
    from activation_exp.runner import main
    main()


def plot():
    from activation_exp.plotting.training_curves import main as tc
    from activation_exp.plotting.dead_neurons import main as dn
    from activation_exp.plotting.activation_dist import main as ad
    tc()
    dn()
    ad()
