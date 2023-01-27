from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pickle
import joblib 
#loaded_model = pickle.load(open('pima_indians_diabetes_model', 'rb'))
loaded_model = joblib.load('pima_indians_diabetes_model.joblib')

# Specify an initial type for the model ( similar to input shape for the model )
initial_type = [ 
    ( 'input_study_hours' , FloatTensorType( [None,1,8] ) ) 
]

# Write the ONNX model to disk
converted_model = convert_sklearn( loaded_model , initial_types=initial_type )
with open( "sklearn_model.onnx", "wb" ) as f:
    f.write( converted_model.SerializeToString() )

