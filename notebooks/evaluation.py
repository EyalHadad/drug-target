from src.models.evaluation.eval_object import EvaluationObj

if __name__ == '__main__':
    train_obj = EvaluationObj()
    target_names = ['ifng', 'kat5', 'tyms', 'dhfr', 'tf', 'pdcd1', 'a2m']
    for _target_data in target_names:
        train_obj.load_and_preprocessing(dest_id=_target_data, is_target=True)
        train_obj.use_model(_model_name="model4")
