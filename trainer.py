from torchsummary import summary
from evaluation import Eval
from model.attention_unet_plusplus import AttUNetPlusPlus
from model.discriminator_metric import *
from utils import *
import logging
import time
import collections

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
        self.generator = AttUNetPlusPlus(
            in_channel=1, ngf=self.opt.ngf).to(self.device_0)
        self.generator.apply(weights_init_normal)
        summary(self.generator, [(2, 256, 256), (1, 256, 256)])
        # self.discriminator_metric = Discriminator_Stride2_SN(
        #     self.opt.ndf).to(self.device_0)
        # self.discriminator_metric.apply(weights_init_normal)
        self.discriminator_metric = NLayerDiscriminator(
            ndf=self.opt.ndf, input_nc=2).to(self.device_0)
        summary(self.discriminator_metric, [(1, 256, 256), (1, 256, 256)])
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=self.opt.lr_G, betas=(0.5, 0.999))
        self.optimizer_D_Metric = torch.optim.Adam(self.discriminator_metric.parameters(), lr=self.opt.lr_D,
                                                   betas=(0.5, 0.999))
        experts = [AttUNetPlusPlus(
            in_channel=1, ngf=self.opt.ngf).to(self.device_0) for i in self.opt.num_experts]
        self.criterion = nn.BCELoss()

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
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer_G, step_size=self.opt.decay_epoch, gamma=0.5)
        scheduler_D_Metric = torch.optim.lr_scheduler.StepLR(
            self.optimizer_D_Metric, step_size=self.opt.decay_epoch, gamma=0.5)
        for epoch in range(self.opt.epochs):
            t0 = time.time()
            epoch += 1
            self.generator.train()
            self.discriminator_metric.train()
            epoch_gen_loss = 0
            epoch_gen_loss_GAN = 0
            epoch_discrim_loss = 0
            epoch_discrim_loss_metric = 0
            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                model = self.train_step(batch)
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
        for epoch in range(self.opt.epochs):
            t0 = time.time()
            epoch += 1
            self.generator.train()
            epoch_gen_loss = 0
            epoch_gen_loss_GAN = 0

            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                model = self.train_step(batch)
                epoch_gen_loss += model.gen_loss.item()
                epoch_gen_loss_GAN += model.gen_loss_GAN.item()

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
