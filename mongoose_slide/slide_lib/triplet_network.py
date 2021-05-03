import torch
import torch.nn as nn
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class TripletNet(nn.Module):
	def __init__(self, margin, K, L, layer_size):
		super(TripletNet, self).__init__()
		self.K = K
		self.L = L
		self.dense1 = nn.Linear(layer_size, K*L)
		self.init_weights(self.dense1.weight, self.dense1.bias)
		self.dense1.bias.requires_grad = False
		self.margin = margin
		self.device = device

	def init_weights(self, weight, bias):
		weight.data.normal_(0,1)
		bias.data.fill_(0)

	def forward(self, arc, pair, label):

		emb_arc = self.dense1(arc)
		emb_pair = self.dense1(pair)

		# embedding chunk - L1
		emb_arc_chunk = torch.cat(torch.chunk(emb_arc, self.L, dim = 1 ))
		emb_pair_chunk = torch.cat(torch.chunk(emb_pair, self.L, dim = 1 ))

		label_chunk = label.repeat(self.L)
		assert emb_arc_chunk.size()==emb_pair_chunk.size()
		assert emb_arc_chunk.size()[0]==label_chunk.size()[0]
		
		beta = 1
		alpha = 0.5
		emb_arc_chunk = torch.tanh( beta * emb_arc_chunk)
		emb_pair_chunk = torch.tanh( beta * emb_pair_chunk)

		# agree_x_p = torch.mean(  ((emb_arc_chunk > 0)  == (emb_pair_chunk > 0 ))[label_chunk==1].float()  )
		# agree_x_n = torch.mean(  ((emb_arc_chunk > 0)  == (emb_pair_chunk > 0 ))[label_chunk==0].float()  )
		#
		# print("\nagree_x_p", agree_x_p)
		# print("agree_x_n", agree_x_n)

		output = torch.sum(emb_arc_chunk * emb_pair_chunk, dim= 1)
		assert output.size()==label_chunk.size()

		output_loss = F.binary_cross_entropy(F.sigmoid(output), label_chunk)
		# reg_loss=torch.mean(torch.norm(self.dense1.weight,dim=1))
		# loss = output_loss+0.05*reg_loss
		loss = output_loss
		return loss

