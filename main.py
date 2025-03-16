import argparse
from q_learning import train_q_learning
from gradient_ascent import train_gradient_ascent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение с подкреплением на примере задачи CartPole.")
    parser.add_argument("--method", type=str, choices=["q_learning", "gradient_ascent"], required=True, help="Метод обучения")
    parser.add_argument("--episodes", type=int, default=10, help="Количество эпизодов для обучения")
    args = parser.parse_args()

    if args.method == "q_learning":
        train_q_learning(episodes=args.episodes)
    elif args.method == "gradient_ascent":
        train_gradient_ascent(episodes=args.episodes)
