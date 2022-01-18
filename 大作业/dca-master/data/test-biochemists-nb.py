import numpy as np
from autoencoder.io import read_text
from autoencoder.network import MLP
from keras.callbacks import TensorBoard

count = read_text('biochemists.tsv', header='infer')
y = count[:, 0].astype(int)
x = count[:, 1:]

net = MLP(x.shape[1], output_size=1, hidden_size=(), masking=False, loss_type='nb')
net.build()
model = net.model
tb = TensorBoard(log_dir='./logs', histogram_freq=1)

model.summary()
model.compile(loss=net.loss, optimizer='Adam')
model.fit(x, y, epochs=700, batch_size=32, callbacks=[tb])


print('Theta: %f' % net.extra_models['dispersion']())
