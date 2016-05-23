import numpy as np

def FourierFeatures(degree, feat):
    if len(feat.shape) == 1:
        num_feat = feat.shape[0]
        new_feat = np.empty( ((degree+1)**num_feat))

        count = np.zeros( (1, num_feat) )
        feat = feat.reshape( (feat.shape[0], 1) )
        for i in range(new_feat.shape[0]):
            new_feat[i] = math.cos(math.pi * count.dot(feat))

            for j in range(count.shape[1]-1, -1, -1):
                if j == count.shape[1]-1:
                    count[0,j] += 1
                elif count[0,j-1] == degree+1:
                    count[0,j-1] = 0
                    count[0,j] += 1
    else:
        num_feat = feat.shape[1]
        new_feat = np.empty( (feat.shape[0], (degree+1)**num_feat))

        for j in range(new_feat.shape[0]):
            count = np.zeros( (1, num_feat) )
            for i in range(new_feat.shape[1]):
                feat_j = feat[j].reshape( (feat[j].shape[0], 1) )
                new_feat[j][i] = math.cos(math.pi * count.dot(feat_j))

                for k in range(count.shape[1]-1, -1, -1):
                    if k == count.shape[1]-1:
                        count[0,k] += 1
                    elif count[0,k-1] == degree+1:
                        count[0,k-1] = 0
                        count[0,k] += 1
    return new_feat


