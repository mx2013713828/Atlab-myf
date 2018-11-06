import yaml

def _init_():
    f = open('pytorch_train.yaml')
    cfg = f.read()

    cfg = yaml.load(cfg)
    # print(cfg)
    return cfg
def main():
    print('--main--')
    cfg = _init_()
    print (cfg)
    print(cfg['Train']['epoch'])
    print(type(cfg['Train']['mean']))
if __name__ == "__main__":
    main()
