# Localization of Memorization in Self-Supervised Learning

*Keywords:* self-supervised learning, memorization, localization

*TL;DR:* We identify where memorization happens in self-supervised learning encoders.

*Abstract:*
Recent work on studying memorization in self-supervised learning (SSL) suggests that even though SSL encoders are trained on millions of images, they still memorize individual data points. While effort has been put into characterizing the memorized data and linking encoder memorization to downstream utility, little is known about where the memorization happens inside SSL encoders. To close this gap, we propose two metrics for localizing memorization in SSL encoders on a per-layer (LayerMem) and per-unit basis (UnitMem). Our localization methods are independent of the downstream task, do not require any label information, and can be performed in a forward pass. By localizing memorization in various encoder architectures (convolutional and transformer-based) trained on diverse datasets with contrastive and non-contrastive SSL frameworks, we find that (1) while SSL memorization increases with layer depth, highly memorizing units are distributed across the entire encoder, (2) a significant fraction of units in SSL encoders experiences surprisingly high memorization of individual data points, which is in contrast to models trained under supervision, (3) *atypical* (or outlier) data points cause much higher layer and unit memorization than standard data points, and (4) in vision transformers, most memorization happens in the fully-connected layers. Finally, we show that localizing memorization in SSL has the potential to improve fine-tuning and to inform pruning strategies.

## Description of the code

Before using the code, install the environments in each model folder first. 

The code mainly contains of the following modules:

1. Candidate model training:
Please first install the requirements.txt for the specific model before use. Then make sure the train_XXX.py is in the same folder with the model files in the model folder. Then modify the datapath, savingpath and other parameters according to your device and experimental needs. The output will be the final model and checkpoints for both candidate and independent encoder.

2. LayerMem meassurement:
Once the candidate and independent model pairs are trained, use memorization_scores.py to test the memorization score for the candidate samples. Make sure the tested candidate samples are the same as the candidate samples used during training. Also modify the datapath, savingpath, model name, and other parameters before using.

3. Linear Probing:
Use linear_probing_train.py to do the linear probing for the pre-trained encoders. The default linear probing classifier is an FC-layer. Make sure to put this file under the same folder with the model_XXX.py as well as the model.pt. Also modify the datapath, savingpath, model name, and other parameters before using.

4. UnitMem and ClassMem:
Use UnitMem.py and ClassMem.py to meassure the UnitMem and ClassMem of pre-trained encoders. Make sure to put this file under the same folder with the model_XXX.py as well as the model.pt. Also modify the datapath, savingpath, model name, and other parameters before using. The default setting for UnitMem is based on the augmentations of SimCLR, always make sure use the same strength and set of augmentations as during training. For ClassMem, you need to generate a list of images to load (you could follow our setting in the paper, pick 1 image per class or 100 images per class). You could either do it with pre-generated list as in our code or generate it yourself. 

5. Locate the Mu_Max:
Use UnitMem_mu_max.py to return the ID of the data points which produced the mu_max of each unit. Make sure to put this file under the same folder with the model_XXX.py as well as the model.pt. Also modify the datapath, savingpath, model name, and other parameters before using.  

6. Layer exchange:
Use layerexchange.py to exchange the specific layer/layers of two models.  Make sure to put this file under the same folder with the model_XXX.py as well as the model.pt. Also modify the datapath, savingpath, model name, and other parameters before using. Always make sure that the exchaged models have same architecture and the exchanged layers have same number of parameters.

7. Single neuron training:
Since pytorch does not support single unit training, use finetune.py to fine-tune the whole layer of the unit. Then use neuron_replace.py to restore the parameters of all units in this layer to their original values, except for the selected unit. Make sure to put this file under the same folder with the model_XXX.py as well as the model.pt. Also modify the datapath, savingpath, model name, and other parameters before using. Always make sure the exchaged models have the same architecture and exchanged layers have the same number of parameters.

8. Neuron pruning:
Use prune.py to prune (zero out all parameters) the units according to three different modes: random, highest UnitMem, and lowest UnitMem. Before using this, you might need to generate a list of the units which have highest/lowest UnitMem in (each layer)/(whole model). You can do it youself or with a pre-generated list. Then you can use the performanceloss.py to evaluate the performance loss of the pruned model, but remember to retrain a new linear probing classifier.

9. Access to intermediate layer ouput:
For well defined models like ResNet, you can use the code like 'new_m = torchvision.models._utils.IntermediateLayerGetter(model,{'layer3_residual2': 'feat1'})' to access the output for the middle layer. While for many other models which do not clearly define the name of every layer, we remove the layers we don't want and use 'model.load_state_dict(torch.load('./name_of_model.pth'), strict=False)' to only load the layers before the layer we selected. For example, the detailed sample of ViT encoder trained with DINO can be found in folder 'DINO/model_modify_for_middle_layer_output'.




