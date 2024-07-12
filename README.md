In the course of drug discovery and combination therapy, drug-drug interactions may results the adverse reactions, affecting not only the treatment of diseases but also exposing new drugs to the risk of delisting. Conventional in vitro and in vivo experiments prove to be arduous and time-consuming in the identification of potential DDIs. While existing computational methods have provided new perspectives for identifying DDIs, they still had limitations. In this paper, we innovatively use the probability transfer matrix combined with Stacked Denoising Auto-encoder proposed a model, which named MultiPT-DDI, to calculate the correlation of edge nodes in the adjacency matrix, so as to effectively learn the multi-level representation of nodes, and reduce the probability bias of marginal nodes in the sparse matrix and the noise of the original data. Specifically, the method firstly used random surfing to sample multiple bipartite graph networks to obtain multiple probability transfer matrices. Subsequently, multiple denoising autoencoder modules are employed for layer-wise unsupervised pre-training of the network. Finally, we inferd the connection between drugs pairs using the Random Forest Classifier. The research results obtained the AUC score of 0.9433 and the AUPR score of 0.9372 in the 5-fold cross-validation, significantly outperforming existing models. In the case studies, 26 of the top 30 drug pairs with the highest scores were validated. The empirical evidence indicate that MultiPT-DDI is an effective complementary model for predicting potential DDIs, providing a reliable reference for traditional experimental methods.
