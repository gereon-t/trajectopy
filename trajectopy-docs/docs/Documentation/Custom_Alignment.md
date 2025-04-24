# Importing a custom alignment	

You can import a custom alignment by importing a custom result file. The result file is a CSV file with the following structure:

```csv
#type Alignment
#name My Custom Alignment
1.23,True
4.56,True
7.89,True
0.00,True
0.10,True
0.20,True
1.00,False
0.00,False
0.00,False
0.00,False
0.00,False
0.00,False
0.00,False
0.00,False
```

The first line of the file is a comment that indicates the type of the file. This line is mandatory. The second line is a comment that indicates the name of the alignment. The rest of the lines are the parameter values followed by a boolean value that indicates whether the parameter is enabled or not. The order of the parameters are:

- `similarity translation x [m]`
- `similarity translation y [m]`
- `similarity translation z [m]`
- `similarity rotation x [rad]`
- `similarity rotation y [rad]`
- `similarity rotation z [rad]`
- `similarity scale x`
- `time shift [s]`
- `leverarm x [m]`
- `leverarm y [m]`
- `leverarm z [m]`
- `sensor rotation x [rad]`
- `sensor rotation y [rad]`
- `sensor rotation z [rad]`

