import numpy as np
from tools.sabra_cure.r2v2_harm import HARM_ORDER,harm_features,wrong,action,ridge,scaler,pred
def test_frozen_harm_feature_order_and_shape():
 x=np.zeros((3,14));f=harm_features(x,np.array([1.,-1.,0.]),np.ones(3));assert len(HARM_ORDER)==22 and f.shape==(3,22)
def test_harm_target_is_wrong_times_abs_y_and_bounded():
 m=np.array([1.,-1.,1.]);y=np.array([.2,.3,-.4]);h=wrong(m,y)*np.abs(y);assert np.allclose(h,[0,.3,.4]) and np.all((h>=0)&(h<=1))
def test_ridge_and_threshold_actions_are_deterministic():
 x=np.arange(12.).reshape(6,2);m,i=scaler(x);b,c=ridge((x-m)/i,np.arange(6.));assert np.allclose(pred(x,m,i,b,c),np.arange(6.),atol=1);assert action(np.array([1.,-1.]),np.array([.2,.3]),.2).tolist()==[1,0]
