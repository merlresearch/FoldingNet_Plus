# Copyright (C) 2018-2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
traintester.py in deepgeom
"""

import os
import sys
import time

import glog as logger
import numpy as np
import scipy.io as sio
import torch
from deepgeom.utils import check_exist_or_mkdirs
from torch.autograd import Variable


class Stats(object):
    def __init__(self):
        self.iter_loss = []

    def push_loss(self, iter, loss):
        self.iter_loss.append([iter, loss])

    def push(self, iter, loss):
        self.push_loss(iter, loss)

    def save(self, file):
        np.savez_compressed(file, iter_loss=np.asarray(self.iter_loss))


class TrainTesterLGAN(object):
    def __init__(self, net, solver, total_epochs, cuda, log_dir, verbose_per_n_batch=1, haar_coeff=0.5, start_epochs=0):
        self.net, self.solver, self.total_epochs, self.cuda = net, solver, total_epochs, cuda
        self.log_dir, self.verbose_per_n_batch = log_dir, verbose_per_n_batch
        check_exist_or_mkdirs(log_dir)
        self.start_epochs = start_epochs
        self.done = False
        self.train_iter = 0
        self.stats_train_batch = Stats()
        self.stats_train_running = Stats()
        self.stats_test = Stats()
        self.running_loss = None
        self.running_factor = 0.9
        self.epoch_callbacks = [self.save_stats]
        self.haar_coeff = haar_coeff

        print("self.haar_coeff: {}".format(self.haar_coeff))
        print("self.start_epochs: {}".format(self.start_epochs))

    def invoke_epoch_callback(self):
        if len(self.epoch_callbacks) > 0:
            for ith, cb in enumerate(self.epoch_callbacks):
                try:
                    cb()
                except:
                    logger.warn("epoch_callback[{}] failed.".format(ith))

    def adjust_lr_linear(self, step, total_step):
        base_lr = self.solver.defaults["lr"]
        lr = base_lr * (total_step - step + 1.0) / total_step
        for param_group in self.solver.param_groups:
            param_group["lr"] = lr

    def rand_rotation_matrix(self):
        all_theta = [0, 90, 180, 270]
        all_axis = np.eye(3)
        theta = all_theta[np.random.randint(0, 4)]

        rotation_theta = np.deg2rad(theta)
        rotation_axis = all_axis[np.random.randint(0, 3), :]
        sin_xyz = np.sin(rotation_theta) * rotation_axis

        R = np.cos(rotation_theta) * np.eye(3)

        R[0, 1] = -sin_xyz[2]
        R[0, 2] = sin_xyz[1]
        R[1, 0] = sin_xyz[2]
        R[1, 2] = -sin_xyz[0]
        R[2, 0] = -sin_xyz[1]
        R[2, 1] = sin_xyz[0]
        R = R + (1 - np.cos(rotation_theta)) * np.dot(
            np.expand_dims(rotation_axis, axis=1), np.expand_dims(rotation_axis, axis=0)
        )
        R = torch.from_numpy(R)
        return R

    def train(self, epoch, loader, loss_fn):
        # import ipdb; ipdb.set_trace()
        self.net.train()
        total_step = self.total_epochs * len(loader)
        finished_step = (epoch - 1) * len(loader)
        loss_sum, batch_loss = 0.0, 0.0
        # print("train begin")
        # print(epoch)
        label_total = []
        batch_start = time.time()

        for batch_idx, batch in enumerate(loader):
            # self.adjust_lr_linear(batch_idx + finished_step, total_step)
            data = batch["data"]  # B_train x N x k ,  k = 3
            num_batch = data.shape[0]
            rotation_matrix = self.rand_rotation_matrix().float()

            if self.cuda:
                data = data.cuda()
                rotation_matrix = rotation_matrix.cuda()
            data = Variable(data)
            rotation_matrix = Variable(rotation_matrix)
            rotation_matrix = (rotation_matrix.unsqueeze(0)).expand(num_batch, -1, -1)

            data = torch.bmm(data, rotation_matrix)
            self.solver.zero_grad()
            rec_pc, mi_loss, global_feature = self.net(data)
            loss = loss_fn(rec_pc, data) + torch.mean(mi_loss)
            loss.backward()
            self.solver.step()

            # loss
            batch_len = len(data)
            batch_loss = loss.item()
            loss_sum += batch_loss
            if self.running_loss is None:
                self.running_loss = batch_loss
            else:
                self.running_loss = self.running_factor * self.running_loss + (1 - self.running_factor) * batch_loss

            # collect stats
            self.train_iter += 1  # epoch x num_train/batch_train
            self.stats_train_batch.push(self.train_iter, loss=batch_loss)
            self.stats_train_running.push(self.train_iter, loss=self.running_loss)

            # logger
            if self.verbose_per_n_batch > 0 and batch_idx % self.verbose_per_n_batch == 0:
                batch_end = time.time()
                logger.info(
                    (
                        "Epoch={:<3d} [{:3.0f}% of {:<5d}] "
                        + "Loss(Batch,Running)={:.3f},{:.3f} "
                        + "; elapsed time: {}"
                    ).format(
                        epoch,
                        100.0 * batch_idx / len(loader),
                        len(loader.dataset),
                        batch_loss,
                        self.running_loss,
                        batch_end - batch_start,
                    )
                )
                batch_start = time.time()

        logger.info(
            "Train set (epoch={:<3d}): Loss(LastBatch,Average)={:.3f},{:.3f}".format(
                epoch, batch_loss, loss_sum / float(len(loader))
            )
        )

    def test(self, loader, loss_fn):
        # volatile=True
        # import ipdb; ipdb.set_trace()
        self.net.eval()
        loss_fn.size_average = False
        test_loss = 0.0
        counter = 0

        label_total = []
        cw_total = []
        rec_pc_total = []
        weight_total = []
        input_total = []
        for batch in loader:
            data = batch["data"]  # B_test x N x k, k = 3
            label = batch["label"]
            if self.cuda:
                data = data.cuda()
            data = Variable(data)  # B_test x N x K
            # _, _, global_feature1 = self.net(data)
            # loss = loss_fn(rec_pc, data) + torch.mean(mi_loss)

            # w_estimate = Variable(self.net(data)[0])
            # w = Variable(self.net(data)[1])
            # output = Variable(torch.bmm(w ,w_estimate))
            #
            # rec_pc = 0.5*output + 0.5*w_estimate

            cw_total.append(np.array(Variable(self.net(data)[2]).data))
            label_total.append(np.array(label))
            # rec_pc_total.append(np.array(rec_pc))
            # weight_total.append(np.array(Variable(self.net(data)[1]).data))
            # input_total.append(np.array(data))

            # test_loss += loss_fn(rec_pc, data)
            test_loss = 0
            counter += 1

        # save the codeword for svm transfer classification
        np.save("MN10_train_latentGAN_e360.npy", {"codewords": cw_total, "label": label_total})
        # test_loss = test_loss.cpu().item() / counter
        self.stats_test.push(self.train_iter, loss=test_loss)
        # logger.info('Test set: AverageLoss={:.4f}'.format(test_loss))
        loss_fn.size_average = True
        # print("test end")

    def save_stats(self):
        self.stats_train_running.save(
            os.path.join(self.log_dir, "stats_train_running_k96_e5_sigma8.npz")
        )  # stats_train_running_k72
        self.stats_train_batch.save(os.path.join(self.log_dir, "stats_train_batch_k96_e5_sigma8.npz"))
        self.stats_test.save(os.path.join(self.log_dir, "stats_test_k48_e5_sigma8.npz"))

    def run(self, train_loader, test_loader, loss_fn, train_or_test):

        logger.check_eq(self.done, False, "Done already!")
        if self.cuda:
            self.net.cuda()

        logger.info("Network Architecture:")
        print(str(self.net))
        sys.stdout.flush()

        logger.info("{} Hyperparameters:".format(self.solver.__class__.__name__))
        print(str(self.solver.defaults))
        sys.stdout.flush()

        if train_or_test == 0:
            for epoch in range(self.start_epochs, self.total_epochs + 1):
                start = time.time()
                self.train(epoch=epoch, loader=train_loader, loss_fn=loss_fn)
                end = time.time()
                print("epoch: {}, Elapsed time: {}".format(epoch, end - start))
                self.invoke_epoch_callback()
                if epoch % 20 == 0:
                    torch.save(self.net.state_dict(), os.path.join(self.log_dir, "parameters_at_{}.pkl".format(epoch)))

        # testing mode
        elif train_or_test == 1:
            print("I am testing!!!!!")
            self.test(loader=test_loader, loss_fn=loss_fn)
            self.invoke_epoch_callback()
        else:
            for epoch in range(self.start_epochs, self.total_epochs + 1):
                self.train(epoch=epoch, loader=train_loader, loss_fn=loss_fn)
                self.test(loader=test_loader, loss_fn=loss_fn)
                self.invoke_epoch_callback()
                if epoch % 20 == 0:
                    torch.save(self.net.state_dict(), os.path.join(self.log_dir, "parameters_at_{}.pkl".format(epoch)))

        self.save_stats()
        self.done = True


class TrainTesterAug(object):
    def __init__(self, net, solver, total_epochs, cuda, log_dir, verbose_per_n_batch=1, haar_coeff=0.5, start_epochs=0):
        self.net, self.solver, self.total_epochs, self.cuda = net, solver, total_epochs, cuda
        self.log_dir, self.verbose_per_n_batch = log_dir, verbose_per_n_batch
        check_exist_or_mkdirs(log_dir)
        self.start_epochs = start_epochs
        self.done = False
        self.train_iter = 0
        self.stats_train_batch = Stats()
        self.stats_train_running = Stats()
        self.stats_test = Stats()
        self.running_loss = None
        self.running_factor = 0.9
        self.epoch_callbacks = [self.save_stats]
        self.haar_coeff = haar_coeff

        print("self.haar_coeff: {}".format(self.haar_coeff))
        print("self.start_epochs: {}".format(self.start_epochs))

    def invoke_epoch_callback(self):
        if len(self.epoch_callbacks) > 0:
            for ith, cb in enumerate(self.epoch_callbacks):
                try:
                    cb()
                except:
                    logger.warn("epoch_callback[{}] failed.".format(ith))

    def adjust_lr_linear(self, step, total_step):
        base_lr = self.solver.defaults["lr"]
        lr = base_lr * (total_step - step + 1.0) / total_step
        for param_group in self.solver.param_groups:
            param_group["lr"] = lr

    def rand_rotation_matrix(self):
        all_theta = [0, 90, 180, 270]
        all_axis = np.eye(3)
        theta = all_theta[np.random.randint(0, 4)]

        rotation_theta = np.deg2rad(theta)
        rotation_axis = all_axis[np.random.randint(0, 3), :]
        sin_xyz = np.sin(rotation_theta) * rotation_axis

        R = np.cos(rotation_theta) * np.eye(3)

        R[0, 1] = -sin_xyz[2]
        R[0, 2] = sin_xyz[1]
        R[1, 0] = sin_xyz[2]
        R[1, 2] = -sin_xyz[0]
        R[2, 0] = -sin_xyz[1]
        R[2, 1] = sin_xyz[0]
        R = R + (1 - np.cos(rotation_theta)) * np.dot(
            np.expand_dims(rotation_axis, axis=1), np.expand_dims(rotation_axis, axis=0)
        )
        R = torch.from_numpy(R)
        return R

    def train(self, epoch, loader, loss_fn):
        # import ipdb; ipdb.set_trace()
        self.net.train()
        total_step = self.total_epochs * len(loader)
        finished_step = (epoch - 1) * len(loader)
        loss_sum, batch_loss = 0.0, 0.0
        # batch_start = time.time()
        for batch_idx, batch in enumerate(loader):
            data = batch["data"]  # B_train x N x k ,  k = 3
            weight1 = batch["augment1"]
            weight2 = batch["augment2"]
            weight3 = batch["augment3"]

            num_batch = data.shape[0]
            rotation_matrix = self.rand_rotation_matrix().float()

            if self.cuda:
                data = data.cuda()
                rotation_matrix = rotation_matrix.cuda()
                weight1 = weight1.cuda()
                weight2 = weight2.cuda()
                weight3 = weight3.cuda()

            rotation_matrix = Variable(rotation_matrix)
            rotation_matrix = (rotation_matrix.unsqueeze(0)).expand(num_batch, -1, -1)

            data = torch.bmm(data, rotation_matrix)
            data_input = torch.cat((data, weight1, weight2, weight3), 2)  # do not rotate the weight matrix
            data_input = Variable(data_input)  # B X 2048 X (3 + 9)
            self.solver.zero_grad()

            coarse_reconstruction, graph, global_feature = self.net(data_input)
            rec_pc = (
                self.haar_coeff * torch.bmm(graph, coarse_reconstruction)
                + (1.0 - self.haar_coeff) * coarse_reconstruction
            )
            #    add lines here to save the intermediate results, e.g. reconstructed point clouds, codewords, etc.
            # if (epoch == 100 or epoch ==1000 or epoch == 3000 or epoch == 5000 or epoch ==10000):
            #   np.save('train_weight_plane_'+str(epoch)+'.npy', {'pc_rec':rec_pc.data.cpu().numpy(), 'pc_weight':output.data.cpu().numpy(), 'input_data':data.cpu().numpy(), 'pc_fold':w_estimate.data.cpu().numpy(), 'weight_matrix':w.data.cpu().numpy()})
            # end
            loss = loss_fn(rec_pc, data)
            loss.backward()
            self.solver.step()
            # loss
            batch_len = len(data)
            batch_loss = loss.item()
            loss_sum += batch_loss
            if self.running_loss is None:
                self.running_loss = batch_loss
            else:
                self.running_loss = self.running_factor * self.running_loss + (1 - self.running_factor) * batch_loss
            #    collect stats
            self.train_iter += 1  # epoch x num_train/batch_train
            self.stats_train_batch.push(self.train_iter, loss=batch_loss)
            self.stats_train_running.push(self.train_iter, loss=self.running_loss)
            #   logger check the duration time
            # if self.verbose_per_n_batch>0 and batch_idx % self.verbose_per_n_batch==0:
            #     batch_end = time.time()
            #     logger.info((
            #         'Epoch={:<3d} [{:3.0f}% of {:<5d}] '+
            #         'Loss(Batch,Running)={:.3f},{:.3f} ' + "; elapsed time: {}").format(
            #         epoch, 100.*batch_idx/len(loader), len(loader.dataset),
            #         batch_loss, self.running_loss, batch_end-batch_start))
            #     batch_start = time.time()

        logger.info(
            "Train set (epoch={:<3d}): Loss(LastBatch,Average)={:.3f},{:.3f}".format(
                epoch, batch_loss, loss_sum / float(len(loader))
            )
        )

    def test(self, loader, loss_fn):

        self.net.eval()
        loss_fn.size_average = False
        test_loss = 0.0
        counter = 0
        label_total = []
        cw_total = []

        for batch in loader:
            data = batch["data"]  # B_test x N x k, k = 3
            label = batch["label"]
            weight = batch["weight"]
            if self.cuda:
                data = data.cuda()
                weight = weight.cuda()
            data_input = torch.cat((data, weight), 2)
            data_input = Variable(data_input)  # B_test x N x K
            #   if you also want to know the reconstructed results
            # _, _, global_feature, _ = self.net(data_input)
            # w_estimate = Variable(self.net(data_input)[0])
            # w = Variable(self.net(data_input)[1])
            # output = Variable(torch.bmm(w ,w_estimate))
            # rec_pc = 0.5*output + 0.5*w_estimate
            # test_loss += loss_fn(rec_pc, data)
            #   save the labels and the codewords
            cw_total.append(np.array(Variable(self.net(data_input)[2]).data))
            label_total.append(np.array(label))
            counter += 1

        # save the codeword for svm transfer classification
        np.save("your_name.npy", {"codewords": cw_total, "label": label_total})
        #    check the loss
        # test_loss = test_loss.cpu().item() / counter
        # self.stats_test.push(self.train_iter, loss=test_loss)
        # logger.info('Test set: AverageLoss={:.4f}'.format(test_loss))
        loss_fn.size_average = True
        # print("test end")

    def save_stats(self):
        self.stats_train_running.save(
            os.path.join(self.log_dir, "stats_train_running_k96_e5_sigma8.npz")
        )  # stats_train_running_k72
        self.stats_train_batch.save(os.path.join(self.log_dir, "stats_train_batch_k96_e5_sigma8.npz"))
        self.stats_test.save(os.path.join(self.log_dir, "stats_test_k48_e5_sigma8.npz"))

    def run_train(self, train_loader, loss_fn, train_or_test):
        print("You are only training!")

        if self.cuda:
            self.net.cuda()

        logger.info("Network Architecture:")
        print(str(self.net))
        sys.stdout.flush()

        logger.info("{} Hyperparameters:".format(self.solver.__class__.__name__))
        print(str(self.solver.defaults))
        sys.stdout.flush()

        # training mode
        for epoch in range(self.start_epochs, self.total_epochs + 1):
            start = time.time()
            self.train(epoch=epoch, loader=train_loader, loss_fn=loss_fn)
            end = time.time()
            print("epoch: {}; Elapsed time: {}".format(epoch, end - start))
            self.invoke_epoch_callback()
            if epoch % 20 == 0:
                torch.save(self.net.state_dict(), os.path.join(self.log_dir, "parameters_at_{}.pkl".format(epoch)))
        self.save_stats()
        self.done = True

    def run_test(self, test_loader, loss_fn, train_or_test):
        print("You are only testing!")
        if self.cuda:
            self.net.cuda()

        logger.info("Network Architecture:")
        print(str(self.net))
        sys.stdout.flush()

        logger.info("{} Hyperparameters:".format(self.solver.__class__.__name__))
        print(str(self.solver.defaults))
        sys.stdout.flush()

        self.test(loader=test_loader, loss_fn=loss_fn)
        self.invoke_epoch_callback()

        self.save_stats()
        self.done = True

    def run(self, train_loader, test_loader, loss_fn, train_or_test):
        try:
            from deepgeom.visualize import make_dot

            y = self.net.forward(Variable(torch.from_numpy(test_loader.dataset[0:2]["data"])))
            g = make_dot(y)
            g.engine = "dot"
            g.format = "pdf"
            print(g.render(filename=os.path.join(self.log_dir, "net.gv")))
        except:
            logger.warn("failed to draw net.")

        logger.check_eq(self.done, False, "Done already!")
        if self.cuda:
            self.net.cuda()

        logger.info("Network Architecture:")
        print(str(self.net))
        sys.stdout.flush()

        logger.info("{} Hyperparameters:".format(self.solver.__class__.__name__))
        print(str(self.solver.defaults))
        sys.stdout.flush()

        # training mode
        if train_or_test == 0:
            for epoch in range(self.start_epochs, self.total_epochs + 1):
                self.train(epoch=epoch, loader=train_loader, loss_fn=loss_fn)
                self.invoke_epoch_callback()
                if epoch % 20 == 0:
                    torch.save(self.net.state_dict(), os.path.join(self.log_dir, "parameters_at_{}.pkl".format(epoch)))

        # testing mode
        elif train_or_test == 1:
            self.test(loader=test_loader, loss_fn=loss_fn)
            self.invoke_epoch_callback()
        else:
            for epoch in range(self.start_epochs, self.total_epochs + 1):
                self.train(epoch=epoch, loader=train_loader, loss_fn=loss_fn)
                self.test(loader=test_loader, loss_fn=loss_fn)
                self.invoke_epoch_callback()
                if epoch % 20 == 0:
                    torch.save(self.net.state_dict(), os.path.join(self.log_dir, "parameters_at_{}.pkl".format(epoch)))

        self.save_stats()
        self.done = True
