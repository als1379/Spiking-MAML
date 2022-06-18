import  torch, os
import  numpy as np
from    omniglotNShot import OmniglotNShot
import  argparse

from    meta import Meta

def main(args):
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('bn', [64]),
        ('if', [args.threshold]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('bn', [64]),
        ('if', [args.threshold]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('bn', [64]),
        ('if', [args.threshold]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('bn', [64]),
        ('if', [args.threshold]),
        ('flatten', []),
        ('linear', [args.n_way, 64]),
        ('if', [args.threshold_last]),
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)
    db_train = OmniglotNShot('omniglot',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)

    maml.load_state_dict(torch.load('./bestmodel'))

    accs = []
    for _ in range(1000//args.task_num):
        # test
        x_spt, y_spt, x_qry, y_qry = db_train.next('test')
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                        torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        # split to single task each time
        # print(list(zip(x_spt, y_spt, x_qry, y_qry)))
        for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
            test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
            accs.append( test_acc )

    # [b, update_step+1]
    accs = np.array(accs).mean(axis=0).astype(np.float16)
    print('Test acc:', accs)
    with open('test.log', 'a') as f:
        f.write('acc:' + str(accs) +'\n')



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--T', type=int, help='neurons time', default=4)
    argparser.add_argument('--threshold', type=float, help='IF neurons threshold', default=1.0)
    argparser.add_argument('--threshold_last', type=float, help='IF last neurons threshold', default=1.0)

    args = argparser.parse_args()

    for x in range(1, 6):
        for th in range(2, 15):
            print(x, th)
            args.threshold = th / 100
            args.threshold_last = th * x / 100
            with open('test.log', 'a') as f:
                f.write('th:' + str(args.threshold) +'\n')
                f.write('th last:' + str(args.threshold_last) +'\n')

            main(args)
