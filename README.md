# HSI_svm_pca_resNet50
use SVM  and PCA_ResNet50 to classify HSI
- stage1:
    use SVM to classify HSI(Hyperspectral Image). opertion : Firstly , transfer 3D data to 2D data , and the origial groundtruth data is 2D data , I transfer it to 1D data. Then, according the quantity of GT(groundTruth) ,  I choose 15 vectors from the first three least and choose 50 vectors from the other vectors. Treat these data as training data.the left data is test data. And I use the NMF as the method of demension reduction . Before NMF , I normalize the data using Z-Score. Then i use NNLS on the W matrix AFTER NMF and get H matrix. Finally i use SVM to fit and predict the label. here is the process and result:
    ![precisionScore](./t2.png)   
