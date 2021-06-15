from src.models.evaluation.eval_object import EvaluationObj

if __name__ == '__main__':
    train_obj = EvaluationObj()
    # eval_brca1_ercc1_bcl2_apaf1
    target_names = ['ifng', 'kat5', 'tyms', 'dhfr', 'tf', 'pdcd1', 'a2m']
    train_obj.load_and_preprocessing(is_target=False)
    train_obj.use_model(_model_name="model3")
    # for _target_data in target_names:
    #     train_obj.load_and_preprocessing(dest_id=_target_data, is_target=True)
    #     train_obj.use_model(_model_name="model4")
