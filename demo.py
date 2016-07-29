
import sys
import math
import logging
import time
import mxnet as mx
import numpy as np
from collections import namedtuple
try:
    import cPickle as pickle
except:
    import pickle
import base64


MyModel = namedtuple("MyModel", ['my_model_exec', 'symbol',
                                 'data', 'label', 'param_blocks', 'static'])
LSTMState = namedtuple("LSTMState", ["c", "h"])


class MXModel(object):

    def __init__(self, xpu=mx.cpu(), *args, **kwargs):
        self.xpu = xpu
        self.loss = None
        #self.args = {}
        #self.args_grad = {}
        #self.args_mult = {}
        #self.auxs = {}
        self.setup(*args, **kwargs)

    def save(self, fname):
        #args_save = {key: v.asnumpy() for key, v in self.args.items()}
        # with open(fname, 'w') as fout:
        #    pickle.dump(args_save, fout)
        pass

    def load(self, fname):
        # with open(fname) as fin:
        #    args_save = pickle.load(fin)
        #    for key, v in args_save.items():
        #        if key in self.args:
        #            self.args[key][:] = v
        pass

    def setup(self, *args, **kwargs):
        raise NotImplementedError("must override this")


class mymodel(MXModel):

    def setup(self, to_predict=1, num_input_picture=3, picture_shape=(100, 100), batchsize=100, num_of_cnn=1, cnn_filter_num_list=[1],
              cnn_kernel_shape_list=[(5, 5)], cnn_pooling_kernel_list=[(2, 2)], num_lstm_layer=2, num_lstm_hidden=3):
        assert(len(cnn_pooling_kernel_list) == num_of_cnn)
        assert(len(cnn_kernel_shape_list) == num_of_cnn)
        assert(len(cnn_filter_num_list) == num_of_cnn)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename='mymodel.log',
                            filemode='a')
        # Todo
        input_shape = {}
        #input_shape['data'] = (batchsize, picture_shape[0], picture_shape[1] * num_input_picture)
        input_shape['data'] = (batchsize, picture_shape[0] * num_input_picture, picture_shape[1])
        output_shape = to_predict * picture_shape[0] * picture_shape[1]
        input_x = mx.sym.Variable('data')  # placeholder for input shape(batchsize * 30000)
        input_y = mx.sym.Variable('regression_label')  # placeholder for output(batchsize * 10000)
        picture_list = mx.sym.SliceChannel(input_x, num_outputs=num_input_picture,
                                           # name="input_slice", axis = 2 )
                                           name="input_slice", axis=1)
        picture_seq_list = []
        # cnn weight need to be set same
        for picture_idx in range(num_input_picture):
            picture = picture_list[picture_idx]
            conv_input = mx.sym.Reshape(data=picture, target_shape=(
                batchsize, 1, picture_shape[0], picture_shape[1]))
            for cnn_idx in range(num_of_cnn):
                conv = mx.sym.Convolution(data=conv_input, kernel=cnn_kernel_shape_list[
                                          cnn_idx], num_filter=cnn_filter_num_list[cnn_idx], name='cnn_encode_' + str(cnn_idx))
                relu = mx.sym.Activation(data=conv, act_type='relu',
                                         name='cnn_encode_act_' + str(cnn_idx))
                pool = mx.sym.Pooling(data=relu, pool_type='max', kernel=cnn_pooling_kernel_list[
                                      0], stride=cnn_pooling_kernel_list[0], name='cnn_encode_pool_' + str(cnn_idx))
                conv_input = pool
            picture_seq = mx.sym.Flatten(conv_input)
            picture_seq_list.append(picture_seq)

        prev_state = []
        prev_state, encoded = lstm_encode(prev_state=prev_state, images=picture_seq_list,
                                          num_lstm_layer=num_lstm_layer, seq_len=num_input_picture, num_hidden=num_lstm_hidden)
        decoded_list = []
        decoded_list = lstm_decode(decoded_list=decoded_list, prev_state=prev_state,
                                   num_lstm_layer=num_lstm_layer, seq_len=to_predict, num_hidden=num_lstm_hidden, dropout=0.)
        outlist = []
        for idx in range(num_lstm_layer):
            input_shape['lstm_init_encode_' + str(idx) + '_c'] = (batchsize, num_lstm_hidden)
        for idx in range(to_predict):
            input_shape['lstm_init_decode_' + str(idx) + '_h'] = (batchsize, num_lstm_hidden)
        for decoded in decoded_list:
            out = mx.sym.FullyConnected(data=decoded,
                                        num_hidden=picture_shape[0] * picture_shape[1],
                                        name="deembeded_2")
            out = mx.sym.Activation(out, act_type="relu")
            outlist.append(out)
        cancat_out = mx.sym.Concat(*outlist, dim=1)
        final_out = mx.sym.LinearRegressionOutput(data=cancat_out, label=input_y, name='regression')
        #final_out = mx.sym.LinearRegressionOutput(data=encoded, label=input_y, name='regression')
        #
        #zip(final_out.list_arguments(), final_out.infer_shape(**input_shape)[0])
        # return final_out, input_shape
        self.out_sym = final_out

        self.input_shape = input_shape

    def prepare(self, ctx=mx.cpu(), initializer=mx.initializer.Uniform(0.1)):
        # my_model, input_shape = self.setup(sentence_size, num_embed, batch_size=batch_size,
        #     vocab_size=vocab_size, dropout=dropout, with_embedding=with_embedding)
        args_names = self.out_sym.list_arguments()
        args_shape, out_shape, aux_shape = self.out_sym.infer_shape(**self.input_shape)
        # print dict(zip(args_names, args_shape))
        args_shape_dict = {}
        for names, shape in zip(args_names, args_shape):
            if names in args_shape_dict:
                if shape == args_shape_dict[names]:
                    continue
                else:
                    print names
            else:
                args_shape_dict[names] = shape
        args_arrays_dict = {}
        args_grad_dict = {}
        for name, shape in args_shape_dict.iteritems():
            args_arrays_dict[name] = mx.nd.zeros(shape, ctx)
            if (name not in ['data', 'regression_label']) and ('init' not in name):
                args_grad_dict[name] = mx.nd.zeros(shape, ctx)
        self.mod_exec = self.out_sym.bind(
            ctx=ctx, args=args_arrays_dict, args_grad=args_grad_dict, grad_req='add')
        param_blocks = []
        idx = 0
        for name, nd_array in args_arrays_dict.iteritems():
            if (name in ['regression_label', 'data']) | ('init' in name):  # input, output
                continue
            initializer(name, args_arrays_dict[name])
            param_blocks.append((idx, args_arrays_dict[name], args_grad_dict[name], name))
            idx += 1
        out_dict = dict(zip(self.out_sym.list_outputs(), self.mod_exec.outputs))
        #print self.mod_exec.outputs
        data = args_arrays_dict['data']
        label = args_arrays_dict['regression_label']
        static = {}
        for name, value in args_arrays_dict.iteritems():
            if 'init' in name:
                static[name] = value
        self.data = data
        self.label = label
        self.param_blocks = param_blocks
        self.static = static
        # return MyModel(my_model_exec = my_model_exec, symbol = my_model, data = data, label = label,
        # param_blocks=param_blocks, static = static)

    def train(self, train_data, train_label,
              dev_data,
              dev_label,
              batch_size,
              optimizer='rmsprop',
              max_grad_norm=5.0,
              learning_rate=0.0005,
              epoch=200,
              eval_func=None):
        if eval_func is None:
            eval_func = lambda x, y: np.std(x.asnumpy() - y.asnumpy())

        #opt = mx.optimizer.create(optimizer)
        #opt.lr = learning_rate

        #updater = mx.optimizer.get_updater(opt)
        #edge = 100
        #self.data = mx.nd.array(np.zeros((100,3*edge,edge)))
            #label = mx.nd.array([images[idx + 4] for idx in range(200, 600)]).reshape((400, edge * edge))
        #self.label = mx.nd.array(np.zeros((100,edge*edge)))
        for iteration in range(epoch):
            tic = time.time()

            for begin in range(0, train_data.shape[0], batch_size):
                batchX = train_data[begin:begin + batch_size]
                batchY = train_label[begin:begin + batch_size]

                if batchX.shape[0] != batch_size:
                    continue

                print "data shape, ", self.data.shape
                print "label shape, ", self.label.shape
                print "input data shape, ", batchX.shape
                print "input label shape, ", batchY.shape
                batchX.copyto(self.data)
                batchY.copyto(self.label)

                self.mod_exec.forward(is_train=True)
                self.mod_exec.backward()

               # train_error = eval_func(self.mod_exec.outputs[0], self.label)
		        train_error = 0

                #norm = 0.0
                #for idx, weight, grad, name in self.param_blocks:
                #    grad /= batch_size
                #    l2_norm = mx.nd.norm(grad).asscalar()
                #    norm += l2_norm * l2_norm

                #norm = math.sqrt(norm)
                for idx, weight, grad, name in self.param_blocks:
                    if norm > max_grad_norm:
                        grad *= (max_grad_norm / norm)
                    updater(idx, grad, weight)
                    grad[:] = 0.0

                #if dev_data is not None and dev_label is not None:
                #    for begin in range(0, dev_data.shape[0], batch_size):
                #        batchX = dev_data[begin:begin + batch_size]
                #        batchY = dev_label[begin:begin + batch_size]

#                        if batchX.shape[0] != batch_size:
#                            continue
#
#                        self.data = batchX
#                        self.label = batchY
#                        self.mod_exec.forward(is_train=False)
#
#                        dev_error = eval_func(self.mod_exec.outputs[0], self.label)
		        dev_error = 0

                #if iteration % 50 == 0 and iteration > 0:
                #    opt.lr *= 0.5
                #    logging.info("reset optimizer learning rate to {}".format(opt.lr))

                toc = time.time()
                train_time = toc - tic
                logging.info("Iter {} Train: Time:{:.3f}s, Training Error:{:.3f} Dev Error:{:.3f}".format(iteration,
                                                                                                          train_time,
                                                                                                          train_error,
                                                                                                          dev_error))
	#if iteration % 50 == 0 & iteration > 0:
    #            train_error = eval_func(self.mod_exec.outputs[0], self.label)
	#	print train_error


def lstm(num_hidden, indata, prev_state, seqidx, layeridx, dropout=0., attr='encode', not_h=False):
    if dropout > 0.0:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                num_hidden=num_hidden * (4 - not_h),
                                name="lstm_l%d_i2h" % (layeridx) + "_" + attr)
    h2h = mx.sym.FullyConnected(data=indata,
                                num_hidden=num_hidden * (4 - not_h),
                                name="lstm_l%d_h2h" % (layeridx) + "_" + attr)

    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=(4 - not_h),
                                      name="t%d_l%d_slice" % (seqidx, layeridx) + "_" + attr)
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    if not not_h:
        out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
        next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    else:
        next_h = None
    return LSTMState(c=next_c, h=next_h)


def lstm_encode(prev_state, images, num_lstm_layer, seq_len, num_hidden, dropout=0.):
    init_LSTMState = []
    for idx in range(num_lstm_layer):
        temp_c = mx.sym.Variable('lstm_init_encode_' + str(idx) + '_c')
        temp_h = mx.sym.Variable('lstm_init_encode_' + str(idx) + '_h')
        init_LSTMState.append(LSTMState(c=temp_c, h=temp_h))
    for seqidx in xrange(seq_len):
        if seqidx == 0:
            prev_state = init_LSTMState
            dp_ratio = 0.0
        else:
            dp_ratio = dropout
        for layeridx in range(num_lstm_layer):
            if layeridx == 0:
                indata = images[seqidx]
            not_h = False
            if layeridx == num_lstm_layer - 1:
                not_h = True
            next_state = lstm(num_hidden=num_hidden, indata=indata, prev_state=prev_state[
                              layeridx], layeridx=layeridx, seqidx=seqidx, not_h=not_h)
            prev_state[layeridx] = next_state
            indata = next_state.h
    out = prev_state[-1].c
    if dropout:
        out = mx.sym.Dropout(data=out, p=dropout)
    return prev_state, out


def lstm_decode(decoded_list, prev_state, num_lstm_layer, seq_len, num_hidden, dropout=0.):
    init_LSTMState = []
    for idx in range(seq_len):
        temp_c = mx.sym.Variable('lstm_init_decode_' + str(idx) + '_c')
        temp_h = mx.sym.Variable('lstm_init_decode_' + str(idx) + '_h')
        init_LSTMState.append(LSTMState(c=temp_c, h=temp_h))
    for seqidx in xrange(seq_len):
        for layeridx in xrange(num_lstm_layer):
            if not layeridx:
                dp_ratio = 0.0
            else:
                dp_ratio = dropout
            if layeridx == 0:
                indata = init_LSTMState[layeridx].h
            next_state = lstm(num_hidden=num_hidden, indata=indata,
                              prev_state=prev_state[layeridx],
                              seqidx=seqidx,
                              layeridx=layeridx,
                              dropout=dp_ratio,
                              attr='decode')
            prev_state[layeridx] = next_state
            indata = next_state.h
            if layeridx == num_lstm_layer - 1:
                out = next_state.h
                if dropout:
                    out = mx.sym.Dropout(data=out, p=dropout)
                decoded_list.append(out)
    return decoded_list


if __name__ == '__main__':
    images = []
    with open('train-data', 'r') as train_data_file:
        for line in train_data_file:
            #print line
            image = pickle.loads(base64.decodestring(line.strip().split('\t')[2]))
            images.append(image)
    edge = len(images[0])
    #edge = 100
    data = mx.nd.array([images[idx] + images[idx + 1] + images[idx + 2] for idx in range(200, 600)])
    #data = mx.nd.array(np.zeros((400,3*edge,edge)))
    label = mx.nd.array([images[idx + 4] for idx in range(200, 600)]).reshape((400, edge * edge))
    #label = mx.nd.array(np.zeros((400,edge*edge)))
    mod = mymodel(picture_shape = (180, 180))
    mod.prepare()
    mod.train(train_data = data, train_label = label, dev_data = data, dev_label = label, batch_size = 100, epoch=10000)
