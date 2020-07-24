from models import train_ae
def main():
    print("Hello World!")
    train_ae.train(train_ae.get_default_hyperparam(),
                   simplified_rating=False,
                   small_dataset=True,
                   load_csv=False,
                   use_mnist=False,
                   loss_user_items_only = True)


if __name__ == "__main__":
    # execute only if run as a script
    main()