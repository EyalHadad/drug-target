from src.models.training.train_object import TrainObj

if __name__ == '__main__':

    train_obj = TrainObj()
    # target_names = ['ifng', 'kat5', 'tyms', 'dhfr', 'tf', 'pdcd1', 'a2m']
    target_names = ['cancer']
    for num in ['3','3','3']:
        train_obj.load_and_preprocessing()
        train_obj.train_model(_model_name="model{0}".format(num))
        # train_obj.predict_local(target_name="cancer", use_existing=False)

    # for _target_data in target_names:
    #     train_obj.load_and_preprocessing(target_data=_target_data)
    #     train_obj.train_model(_model_name="model4")
    #     train_obj.predict_local(target_name="cancer", use_existing=False)
