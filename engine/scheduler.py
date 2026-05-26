import math


def get_lr(round_num: int, config) -> float:
    base   = config.learning_rate
    min_lr = getattr(config, "lr_min", 0.0)
    T      = config.num_rounds
    t      = round_num
    name   = getattr(config, "lr_scheduler", "constant").lower()

    if name == "constant":
        return base

    if name == "cosine":
        return min_lr + (base - min_lr) * 0.5 * (1.0 + math.cos(math.pi * t / max(T - 1, 1)))

    if name == "step":
        return max(min_lr, base * (getattr(config, "lr_gamma", 0.5) ** (t // max(1, getattr(config, "lr_step_size", 20)))))

    if name == "warmup_cosine":
        warmup = getattr(config, "lr_warmup_rounds", 5)
        if t < warmup:
            return base * (t + 1) / max(warmup, 1)
        progress = (t - warmup) / max(T - warmup - 1, 1)
        return min_lr + (base - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    if name == "exponential":
        return max(min_lr, base * (getattr(config, "lr_gamma", 0.95) ** t))

    if name == "polynomial":
        return min_lr + (base - min_lr) * max(0.0, 1.0 - t / max(T, 1)) ** getattr(config, "lr_power", 1.0)

    return base
