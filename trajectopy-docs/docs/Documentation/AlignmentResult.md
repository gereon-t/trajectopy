
## <kbd>module</kbd> `trajectopy.core.alignment.result`
Trajectopy - Trajectory Evaluation in Python 

Gereon Tombrink, 2025 tombrink@igg.uni-bonn.de 



---

### <kbd>class</kbd> `AlignmentResult`
Class to store the result of an alignment. 



**Attributes:**
 
 - <b>`name`</b> (str):  Name of the result. 
 - <b>`position_parameters`</b> (AlignmentParameters):  Estimated position alignment parameters. 
 - <b>`rotation_parameters`</b> (SensorRotationParameters):  Estimated rotation alignment parameters. 
 - <b>`estimation_of`</b> (AlignmentEstimationSettings):  Settings defining which parameters were estimated. 
 - <b>`converged`</b> (bool):  Whether the estimation converged. 

### <kbd>method</kbd> `AlignmentResult.__init__`

```python
__init__(
    name: str = 'Alignment Result',
    position_parameters: trajectopy.core.alignment.parameters.AlignmentParameters = <factory>,
    rotation_parameters: trajectopy.core.alignment.parameters.SensorRotationParameters = <factory>,
    estimation_of: trajectopy.settings.AlignmentEstimationSettings = <factory>,
    converged: bool = True
) → None
```








---

#### <kbd>classmethod</kbd> `AlignmentResult.from_file`

```python
from_file(filename: str) → AlignmentResult
```

Load the result from a file. 



**Args:**
 
 - <b>`filename`</b> (str):  Path to the file. 



**Returns:**
 
 - <b>`AlignmentResult`</b>:  The loaded result. 

---

#### <kbd>method</kbd> `AlignmentResult.to_file`

```python
to_file(filename: str) → None
```

Save the result to a file. 



**Args:**
 
 - <b>`filename`</b> (str):  Path to the file. 


