import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
from    copy import deepcopy
from learner import Learner


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.T=args.T
        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr, betas=(0.8, 0.99), eps=1e-8,
                 weight_decay=1e-6, amsgrad=False)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter


    def forward(self, db_train, device):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        losses_q = [0 for _ in range(self.update_step)]
        corrects = [0 for _ in range(self.update_step)]
        w0 = deepcopy(self.state_dict())
        w_last = deepcopy(self.state_dict())
        for ii in range(4):
            print(type(w0))

            self.load_state_dict(w0)
            
            x_spt, y_spt, x_qry, y_qry = db_train.next()
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
            task_num, setsz, c_, h, w = x_spt.size()
            querysz = x_qry.size(1)
            T=self.T

            for i in range(task_num):
                w0 = self.net.parameters()
                x = x_spt[i]
                y = y_spt[i]
                x_q = x_qry[i]
                y_q = y_qry[i]
                fast_weights = self.net.parameters()
                for k in range(0, self.update_step):
                    logits_q, fast_weights = self.network(self.net, fast_weights, x, x_q, y)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    loss_q = F.cross_entropy(logits_q, y_q)
                    losses_q[k] += loss_q
                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                        corrects[k] = corloss_q = losses_q[-1] / task_numrects[k] + correct
            # end of all tasks
            # sum over all losses on query set across all tasks
            loss_q = losses_q[-1] / task_num

            # optimize theta parameters
            self.meta_optim.zero_grad()
            loss_q.backward()
            self.load_state_dict(w_last)
            self.meta_optim.step()
            w_last = self.state_dict()


        accs = np.array(corrects) / (querysz * task_num)

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)
        corrects = [0 for _ in range(self.update_step_test)]
        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        x = x_spt
        y = y_spt
        x_q = x_qry
        y_q = y_qry
        fast_weights = self.net.parameters()
        for k in range(0, self.update_step_test):
            logits_q, fast_weights = self.network(net, fast_weights, x, x_q, y)
            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                # self.correct_wrong_details(logits_q, y_qry, pred_q)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[k] = corrects[k] + correct
        del net

        accs = np.array(corrects) / querysz

        return accs

    def network(self, net, fast_weights, x, x_q, y):
        # 1. run the i-th task and compute loss
        logits = net(x, self.T, fast_weights, bn_training=True)
        loss = F.cross_entropy(logits, y)
        # 2. compute grad on theta_pi
        grad = torch.autograd.grad(loss, fast_weights)
        # 3. theta_pi = theta_pi - train_lr * grad
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
        logits_q = net(x_q, self.T, fast_weights, bn_training=True)

        return logits_q, fast_weights

    def correct_wrong_details(self, logits_q, y_qry, pred_q):
        correct = 0
        correct_correct = 0
        wrong_correct = 0
        correct_wrong_zero = 0
        wrong_wrong_zero = 0
        correct_wrong_two = 0
        wrong_wrong_two = 0
        wrong_wrong_two_correct_is_in_result = 0
        for i in range(len(logits_q)):
            if logits_q[i].sum() == 1:
                if y_qry[i] == pred_q[i]:
                    correct += 1
                    correct_correct +=1
                else:
                    wrong_correct += 1
            if logits_q[i].sum() == 0:
                if y_qry[i] == pred_q[i]:
                    correct += 1
                    correct_wrong_zero += 1
                else:
                    wrong_wrong_zero += 1
            if logits_q[i].sum() > 1:
                if y_qry[i] == pred_q[i]:
                    correct += 1
                    correct_wrong_two += 1
                else:
                    wrong_wrong_two += 1
                    if logits_q[i][pred_q[i]] == 1:
                        wrong_wrong_two_correct_is_in_result += 1
        print('correct:', correct)
        print('correct_correct:', correct_correct)
        print('wrong_correct:', wrong_correct)
        print('correct_wrong_zero:', correct_wrong_zero)
        print('wrong_wrong_zero:', wrong_wrong_zero)
        print('correct_wrong_two:', correct_wrong_two)
        print('wrong_wrong_two:', wrong_wrong_two)
        print('wrong_wrong_two_correct_is_in_result:', wrong_wrong_two_correct_is_in_result)

    