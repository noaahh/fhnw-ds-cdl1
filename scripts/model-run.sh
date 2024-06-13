python /teamspace/studios/this_studio/cdl1-sensor-based/src/train.py experiment=bidir_lstm
python /teamspace/studios/this_studio/cdl1-sensor-based/src/train.py experiment=cnn
python /teamspace/studios/this_studio/cdl1-sensor-based/src/train.py experiment=log_reg
python /teamspace/studios/this_studio/cdl1-sensor-based/src/train.py experiment=lstm
python /teamspace/studios/this_studio/cdl1-sensor-based/src/train.py experiment=transformer
python /teamspace/studios/this_studio/cdl1-sensor-based/src/train.py experiment=x_lstm

python /teamspace/studios/this_studio/cdl1-sensor-based/src/train.py experiment=bidir_lstm refit_on_all_data=True
python /teamspace/studios/this_studio/cdl1-sensor-based/src/train.py experiment=cnn refit_on_all_data=True
python /teamspace/studios/this_studio/cdl1-sensor-based/src/train.py experiment=log_reg refit_on_all_data=True
python /teamspace/studios/this_studio/cdl1-sensor-based/src/train.py experiment=lstm refit_on_all_data=True
python /teamspace/studios/this_studio/cdl1-sensor-based/src/train.py experiment=transformer refit_on_all_data=True
python /teamspace/studios/this_studio/cdl1-sensor-based/src/train.py experiment=x_lstm refit_on_all_data=True