from automsr.automsr_manager.pathway_gnn_manager import PathWayGnnManager
from pathway_utils import PathWay

gcn_architecture = ["const", "sum", 2, 64,  "tanh", "cos", "mean", 1, 64, "softplus"]

train_epoch = 100
learning_rate = 0.01
learning_rate_decay = 0
min_batch = 15
embedding_dim = 256

print("GNN Architecture:", str(gcn_architecture), "\n")

print("Hyper Parameters:\n"
      "Learning rate: %f\n"
      "Regularization Strength: %f\n"
      "Mini Batch Size: %f\n"
      "Embedding Dimension: %f"%
      (learning_rate,
       learning_rate_decay,
       min_batch,
       embedding_dim))

data = PathWay(0.8, 0.1, min_batch, embedding_dim)

model = PathWayGnnManager(drop_out=0.0,
                          learning_rate=learning_rate,
                          learning_rate_decay=learning_rate_decay,
                          train_epoch=train_epoch)

model.build_model(gcn_architecture, data)

model.train(test_mode=True)

