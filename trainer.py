from torchsummary import summary
from evaluation import Eval
from model.attention_unet_plusplus import AttUNetPlusPlus
from model.discriminator_metric import *
from utils import *
import logging
import time
import collections
from torch.utils.tensorboard import SummaryWriter
import torchvision

Model = collections.namedtuple("Model",
                               'discrim_loss, discrim_loss_metric, gen_loss_GAN, gen_loss_L1, gen_loss,'
                               'outputs, ISTFT_loss, input_to_check, perceptual_loss, val_output, val_target, phase_loss')

TestOutput = collections.namedtuple("TestOutput",
                                    "gen_loss_GAN, perceptual_loss, ISTFT_loss, gen_loss_L1, gen_loss, discrim_loss, "
                                    "metrics_list, discrim_loss_metric")


class Trainer:
    def __init__(self, train_ds, test_ds, device, opt):
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.device_0 = device
        # self.device_1 = device[1]
        self.opt = opt
        # self.generator = AttUNetPlusPlus(
        #     in_channel=1, ngf=self.opt.ngf).to(self.device_0)
        # self.generator.apply(weights_init_normal)
        # self.discriminator_metric = Discriminator_Stride2_SN(
        #     self.opt.ndf).to(self.device_0)
        # self.discriminator_metric.apply(weights_init_normal)
        self.discriminator_metric = NLayerDiscriminator(
            ndf=self.opt.ndf, input_nc=2).to(self.device_0)
        summary(self.discriminator_metric, [(1, 256, 256), (1, 256, 256)])
        # self.optimizer_G = torch.optim.Adam(
        #     self.generator.parameters(), lr=self.opt.lr_G, betas=(0.5, 0.999))
        self.optimizer_D_Metric = torch.optim.Adam(self.discriminator_metric.parameters(), lr=self.opt.lr_D,
                                                   betas=(0.5, 0.999))
        self.experts = [AttUNetPlusPlus(
            in_channel=1, ngf=self.opt.ngf).to(self.device_0).apply(weights_init_normal) for i in range(self.opt.num_experts)]
        summary(self.experts[0], [(2, 256, 256), (1, 256, 256)])

        self.expert_optimizers = [torch.optim.Adam(
            self.experts[i].parameters(), lr=self.opt.lr_G, betas=(0.5, 0.999))for i in range(self.opt.num_experts)]
        self.criterion = nn.BCELoss()
        self.L2loss = torch.nn.MSELoss(reduction='mean')

    def train_step_experts(self, batch, writers):
        noisy_stft = batch[1].to(self.device_0)
        clean_stft_mag = batch[4].to(self.device_0)
        noisy_stft_mag = batch[5].to(self.device_0)
        noise_type = batch[6]
        print(noise_type)

        # 4) Scores D(E_i(X)) from D for all outputs from the experts (p) :
        exp_scores = []  # [num_experts,batch_size, 1 , 16,16]
        exp_scores_mean = []  # [num_experts, batch_size, 1]
        exp_outputs = []  # [num_experts,batch_size, 1 , 256,256]
        for expert in self.experts:
            # [batch_size, 1, 256,256]
            exp_output = expert(noisy_stft, noisy_stft_mag)
            exp_outputs.append(exp_output)
            exp_score = self.discriminator_metric(
                clean_stft_mag, exp_output)  # [batch_size, 1 , 16,16]
            exp_scores.append(exp_score)
            # calculate average expert score over all discriminator patches
            exp_score_mean = torch.mean(exp_score, (2, 3))  # [batch_size, 1]
            exp_scores_mean.append(exp_score_mean)

        # # Log expert scores
        # for sample_idx, transf_idx in enumerate(transformation_idx):
        #     for i in range(len(exp_scores)):
        #         score = exp_scores[i].squeeze()[sample_idx].item()
        #         writers[i].add_scalar(
        #             f'D(E(X))__{transformation_dict[transf_idx]}', score, global_step=total_step)

        # ----------------------------------------------------------------------------------------------------------------
        # 6) Train experts
        exp_outputs_cat = torch.cat(exp_outputs, dim=1)
        exp_scores_mean_cat = torch.cat(exp_scores_mean, dim=1)
        mask_winners = exp_scores_mean_cat.argmax(dim=1)
        # Update each expert on samples it won
        for i, expert in enumerate(self.experts):
            winning_indexes = mask_winners.eq(i).nonzero().squeeze(dim=-1)
            n_expert_samples = winning_indexes.size(0)

            if n_expert_samples > 0:
                exp_samples = exp_outputs_cat[winning_indexes, i].unsqueeze(
                    dim=1)
                D_E_x_transf = exp_scores[i][winning_indexes]
                # D_E_x_transf_ = discriminator(exp_samples.detach())

                loss_E = self.criterion(
                    D_E_x_transf, torch.ones_like(D_E_x_transf))
                self.expert_optimizers[i].zero_grad()
                # TODO figure out why retain graph is necessary
                # loss_E.backward(retain_graph=True)
                loss_E.backward()
                self.expert_optimizers[i].step()
                noisy_references = noisy_stft_mag[winning_indexes]
                exp_samples_grid = torchvision.utils.make_grid(exp_samples)
                noisy_references_grid = torchvision.utils.make_grid(
                    noisy_references)
                writers[i].add_scalar(
                    f'Loss_for_won_samples', loss_E, global_step=total_step)
                writers[i].add_image(
                    f'Expert_{i}_won_transformed_samples', exp_samples_grid, global_step=total_step)
                writers[i].add_image(
                    f'Expert_{i}_won_noise_samples', noisy_references_grid, global_step=total_step)
        # ----------------------------------------------------------------------------------------------------------------

    def train_step(self, batch):
        clean_stft = batch[0].to(self.device_0)
        noisy_stft = batch[1].to(self.device_0)
        clean_audio = batch[2].to(self.device_0)
        noisy_audio = batch[3].to(self.device_0)
        clean_stft_mag = batch[4].to(self.device_0)
        noisy_stft_mag = batch[5].to(self.device_0)

        one_labels = torch.ones(self.opt.batch_size, device=self.device_0)
        # out_mag, out_complex, _ = self.generator(
        #     noisy_stft, noisy_stft_mag)
        # out_mag, _, _ = self.generator(noisy_stft, noisy_stft_mag)

        out_mag = self.generator(noisy_stft, noisy_stft_mag)
        self.optimizer_G.zero_grad()
        # phase_loss = torch.mean(torch.abs(clean_stft - out_complex))
        # Generator loss
        '''Calculate the new istft loss for the batched Datapipeline'''
        # ISTFT_loss, val_output, val_target = \
        #     calculate_loss(out_mag, clean_audio, out_complex,
        #                    self.opt.threshold, self.device_0, self.opt.signal_minmax)

        predict_fake_metric = self.discriminator_metric(
            clean_stft_mag, out_mag)
        # gen_loss_GAN = F.mse_loss(
        #     predict_fake_metric.flatten(), one_labels.float())

        gen_loss_GAN = self.criterion(
            predict_fake_metric, torch.ones_like(predict_fake_metric).to(self.device_0))
        gen_loss_L1 = torch.mean(torch.abs(clean_stft_mag - out_mag))

        # gen_loss = gen_loss_GAN * self.opt.weights[0] + gen_loss_L1 * self.opt.weights[1] + ISTFT_loss * self.opt.weights[2] +\
        #     self.opt.weights[3] * phase_loss
        gen_loss = gen_loss_GAN + gen_loss_L1

        gen_loss.backward()
        self.optimizer_G.step()
        # Discriminator Metric training
        # pesq_score = batch_pesq(val_target, val_output, self.device_0,
        #                         self.opt.pesq_mode)
        # if pesq_score is not None:
        #     self.optimizer_D_Metric.zero_grad()
        #     predict_fake_metric = self.discriminator_metric(
        #         clean_stft_mag, out_mag.detach())
        #     predict_real_metric = self.discriminator_metric(
        #         clean_stft_mag, clean_stft_mag)
        #     discrim_loss_metric = self.opt.weights[0] * (F.mse_loss(predict_real_metric.flatten(), one_labels) +
        #                                                  F.mse_loss(predict_fake_metric.flatten(), pesq_score))
        #     discrim_loss_metric.backward()
        #     self.optimizer_D_Metric.step()
        # else:
        #     discrim_loss_metric = torch.tensor([0.])

        # ---------------------
        self.optimizer_D_Metric.zero_grad()
        predict_fake_metric = self.discriminator_metric(
            noisy_stft_mag, out_mag.detach())
        predict_real_metric = self.discriminator_metric(
            noisy_stft_mag, clean_stft_mag)
        loss_D_real = self.criterion(predict_real_metric, torch.ones_like(
            predict_real_metric).to(self.device_0))
        loss_D_fake = self.criterion(predict_fake_metric, torch.zeros_like(
            predict_fake_metric).to(self.device_0))
        discrim_loss_metric = loss_D_real + loss_D_fake
        discrim_loss_metric.backward()
        self.optimizer_D_Metric.step()
        # ---------------------
        ISTFT_loss = torch.tensor([1., -1.])
        val_output = torch.tensor([1., -1.])
        val_target = torch.tensor([1., -1.])
        phase_loss = torch.tensor([1., -1.])
        return Model(
            discrim_loss=torch.mean(discrim_loss_metric),
            discrim_loss_metric=torch.mean(discrim_loss_metric),
            gen_loss_GAN=torch.mean(gen_loss_GAN),
            gen_loss_L1=torch.mean(gen_loss_L1),
            gen_loss=torch.mean(gen_loss),
            outputs=out_mag,
            ISTFT_loss=torch.mean(ISTFT_loss),
            input_to_check=noisy_audio,
            perceptual_loss=torch.mean(ISTFT_loss),
            val_output=val_output,
            val_target=val_target,
            phase_loss=torch.mean(phase_loss)
        )

    def test_step(self, batch, metrics_list):
        clean_stft = batch[0].to(self.device_0)
        noisy_stft = batch[1].to(self.device_0)
        clean_audio = batch[2].to(self.device_0)
        clean_stft_mag = batch[4].to(self.device_0)
        noisy_stft_mag = batch[5].to(self.device_0)
        one_labels = torch.ones(self.opt.batch_size, device=self.device_0)

        out_mag, out_complex, _ = self.generator(
            noisy_stft, noisy_stft_mag)
        phase_loss = torch.mean(torch.abs(out_complex - clean_stft))

        ISTFT_loss, val_output, val_target = \
            calculate_loss(out_mag, clean_audio, out_complex,
                           self.opt.threshold, self.device_0, self.opt.signal_minmax)

        predict_fake_metric = self.discriminator_metric(
            clean_stft_mag, out_mag.detach())
        predict_real_metric = self.discriminator_metric(
            clean_stft_mag, clean_stft_mag)
        pesq_score = batch_pesq(val_target, val_output)
        if pesq_score is not None:
            discrim_loss_metric = F.mse_loss(predict_real_metric.flatten(), one_labels) + \
                F.mse_loss(predict_fake_metric.flatten(), pesq_score)
        else:
            discrim_loss_metric = torch.tensor([0.])

        gen_loss_GAN = F.mse_loss(
            predict_fake_metric.flatten(), one_labels.float())
        gen_loss_L1 = torch.mean(torch.abs(clean_stft_mag - out_mag))
        gen_loss = gen_loss_GAN * self.opt.weights[0] + gen_loss_L1 * self.opt.weights[1] + ISTFT_loss * self.opt.weights[2] + \
            self.opt.weights[3] * phase_loss

        metrics_score = Parallel(n_jobs=-1)(delayed(metrics)(c, n)
                                            for c, n in zip(val_target, val_output))
        metrics_score = np.array(metrics_score)
        metrics_list_val = np.sum(metrics_score, axis=0)
        metrics_list += metrics_list_val

        return TestOutput(
            gen_loss_GAN=torch.mean(gen_loss_GAN),
            perceptual_loss=torch.mean(ISTFT_loss),
            ISTFT_loss=torch.mean(ISTFT_loss),
            gen_loss_L1=torch.mean(gen_loss_L1),
            gen_loss=gen_loss,
            discrim_loss=torch.mean(discrim_loss_metric),
            discrim_loss_metric=torch.mean(discrim_loss_metric),
            metrics_list=metrics_list
        )

    def test(self):
        metrics_list = np.zeros(6)
        test_gen_loss = 0
        test_discrim_loss = 0
        test_discrim_loss_metric = 0
        self.generator.eval()
        self.discriminator_metric.eval()
        with torch.no_grad():
            for index, batch in enumerate(self.test_ds):
                step_test = index + 1
                test_output = self.test_step(batch, metrics_list)
                test_gen_loss += test_output.gen_loss.item()
                test_discrim_loss += test_output.discrim_loss.item()
                test_discrim_loss_metric += test_output.discrim_loss_metric.item()

        metrics_list = test_output.metrics_list / \
            (step_test * self.opt.batch_size)
        mean_pesq = metrics_list[0]
        mean_csig = metrics_list[1]
        mean_cbak = metrics_list[2]
        mean_covl = metrics_list[3]
        mean_ssnr = metrics_list[4]
        mean_stoi = metrics_list[5]

        test_gen_loss = test_gen_loss/step_test
        test_discrim_loss = test_discrim_loss/step_test
        template = '\n gen_loss: {}, discrim_loss: {}, pesq: {}, csig: {}, ' \
                   'cbak: {}, covl: {}, ssnr: {}, stoi: {} \n'
        logging.info(
            template.format(test_gen_loss, test_discrim_loss, mean_pesq, mean_csig,
                            mean_cbak, mean_covl, mean_ssnr, mean_stoi))

    def train(self):
        writers = []
        log_dir_exp = f'{self.opt.log_dir}/{self.opt.name}'
        os.mkdir(log_dir_exp)
        for i in range(self.opt.num_experts):
            writer_dir = f'{log_dir_exp}/expert{i}'
            os.mkdir(writer_dir)
            writers.append(SummaryWriter(log_dir=writer_dir))
        discriminator_log_dir = f"{log_dir_exp}/discriminator"
        os.mkdir(discriminator_log_dir)
        writers.append(SummaryWriter(log_dir=discriminator_log_dir))

        expert_schedulers = [torch.optim.lr_scheduler.StepLR(
            self.expert_optimizers[i], step_size=self.opt.decay_epoch, gamma=0.5)for i in range(self.opt.num_experts)]
        # scheduler_G = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer_G, step_size=self.opt.decay_epoch, gamma=0.5)
        scheduler_D_Metric = torch.optim.lr_scheduler.StepLR(
            self.optimizer_D_Metric, step_size=self.opt.decay_epoch, gamma=0.5)

        # TODO load experts
        # for i, expert in enumerate(self.experts):
        #     model_path = f"{self.opt.checkpt_dir}/{self.opt.model_for_init_experts}_E_{i}_init.pt"
        #     expert.load_state_dict(torch.load(model_path))

        for epoch in range(self.opt.epochs):
            t0 = time.time()
            epoch += 1
            # self.generator.train()

            for expert in self.experts:
                expert.train()

            self.discriminator_metric.train()
            epoch_gen_loss = 0
            epoch_gen_loss_GAN = 0
            epoch_discrim_loss = 0
            epoch_discrim_loss_metric = 0
            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                model = self.train_step_experts(batch, writers)
                epoch_gen_loss += model.gen_loss.item()
                epoch_gen_loss_GAN += model.gen_loss_GAN.item()
                epoch_discrim_loss += model.discrim_loss.item()
                epoch_discrim_loss_metric += model.discrim_loss_metric.item()

                if (step % self.opt.log_interval) == 0:
                    template = 'Epoch {}, Step {}, gen_loss: {}, gen_loss_GAN: {}, ISTFT_loss: {}, discrim_loss: {}, ' \
                               'discrim_loss_metric: {}, perceptual_loss: {}, gen_loss_L1: {}, phase_loss: {}'
                    logging.info(template.format(epoch, step,
                                                 model.gen_loss.item(),
                                                 model.gen_loss_GAN.item(),
                                                 model.ISTFT_loss.item(),
                                                 model.discrim_loss.item(),
                                                 model.discrim_loss_metric.item(),
                                                 model.perceptual_loss.item(),
                                                 model.gen_loss_L1.item(),
                                                 model.phase_loss.item()))

            template_2 = 'training time: {} seconds, epoch gen loss: {}, epoch gen loss GAN: {}, ' \
                         'epoch discrim loss: {}, epoch discrmi loss metric: {}'
            logging.info(template_2.format(
                time.time() - t0,
                epoch_gen_loss/step,
                epoch_gen_loss_GAN/step,
                epoch_discrim_loss/step,
                epoch_discrim_loss_metric/step
            ))
            if epoch == 1:
                self.test()
            if epoch % 3 == 0:
                self.test()
            if epoch >= 12 and epoch % 3 == 0:
                path = self.opt.save_model + '_own_' + str(epoch)
                torch.save(self.generator.state_dict(), path)
            scheduler_G.step()
            scheduler_D_Metric.step()

    def initialize_experts(self):
        log_dir_exp = f'{self.opt.log_dir}/{self.opt.name}_init'
        os.mkdir(log_dir_exp)
        for i, expert in enumerate(self.experts):
            writer_dir = f'{log_dir_exp}/expert{i}'
            os.mkdir(writer_dir)
            writer = SummaryWriter(log_dir=writer_dir)
            expert.train()
            for epoch in range(self.opt.epochs):
                t0 = time.time()
                epoch += 1
                epoch_gen_loss = 0
                epoch_size = len(self.train_ds)
                for step, batch in enumerate(self.train_ds):
                    step += 1
                    total_step = step + (epoch-1) * epoch_size

                    #########################################################
                    noisy_stft_mag = batch[5].to(self.device_0)
                    out_mag = expert(noisy_stft_mag, noisy_stft_mag)
                    loss = self.L2loss(out_mag, noisy_stft_mag)
                    self.expert_optimizers[i].zero_grad()
                    loss.backward()
                    self.expert_optimizers[i].step()
                    #########################################################
                    loss = torch.mean(loss)
                    epoch_gen_loss += loss

                    if (total_step % self.opt.log_interval) == 0:
                        template = 'Epoch {}, Step {}, expertLoss: {}'
                        logging.info(template.format(epoch, step, loss))
                        writer.add_scalar(
                            f"Initialization_loss_per_step", loss, global_step=total_step)

                        noisy_stft_mag = torch.div(
                            torch.add(noisy_stft_mag, 1), 2)
                        out_mag = torch.div(torch.add(out_mag, 1), 2)
                        img_grid_fake = torchvision.utils.make_grid(out_mag)
                        img_grid_canonical = torchvision.utils.make_grid(
                            noisy_stft_mag)

                        writer.add_image(
                            f"Expert{i} Generated mag", img_grid_fake, global_step=total_step
                        )
                        writer.add_image(
                            f"Expert{i} Corresponding canonical mag", img_grid_canonical, global_step=total_step
                        )

                template_2 = 'training time: {} seconds, epoch gen loss: {}'
                logging.info(template_2.format(
                    time.time() - t0,
                    epoch_gen_loss/total_step
                ))

                writer.add_scalar(
                    f"Initialization_loss_per_epoch", epoch_gen_loss/total_step, global_step=epoch)

                path = os.path.join(self.opt.checkpt_dir,
                                    f"{self.opt.name}_E_{i}_init.pt")
                torch.save(expert.state_dict(), path)
