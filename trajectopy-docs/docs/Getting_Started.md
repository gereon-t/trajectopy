## Installation (with GUI)

It is recommended to install trajectopy with the GUI using the following command:

```console
pip install "trajectopy[gui]"
```

Run
  
```console
trajectopy
```

## Installation (without GUI)

To install trajectopy without the GUI, use the following command:

```console
pip install trajectopy
```

Now you can use trajectopy as a Python package in your scripts.

## Command Line Options (GUI version only)
```console	
usage: trajectopy [-h] [--version] [--single_thread] [--report_settings REPORT_SETTINGS] [--report_path REPORT_PATH]
```
  
```console
options:
    -h, --help            show this help message and exit
    --version, -v
    --single_thread       Disable multithreading
    --report_settings REPORT_SETTINGS, -s REPORT_SETTINGS
                            Path to JSON report settings file
                            that will override the default settings.
    --report_path REPORT_PATH, -o REPORT_PATH
                            Output directory for all reports of one session. If not specified, a temporary directory will be used.
    --mapbox_token MAPBOX_TOKEN, -t MAPBOX_TOKEN
                            Mapbox token to use Mapbox map styles in trajectory plots.
```
Trajectopy allows users to customize the report output path and settings. By default, reports are stored in a temporary directory that will be deleted when the program exits. If you want to keep the reports, you can specify a custom output path using the `--report_path` option. The report settings can be customized using a JSON file.
The report settings file must include all available settings. You can find a sample file [here](https://github.com/gereon-t/trajectopy/blob/main/example_data/custom.json).

Example:
```console
trajectopy -s custom.json -o ./persistent_report_directory
```


## Importing Trajectories

Trajectories can be imported using the "Add" button below the trajectory table or by dragging files into the area of the trajectory table.
Trajectory files must be ASCII files with a csv-like layout, by default, trajectopy filters for the ".traj" extension. The default column structure that can be read without any configuration is the following:

| time | position x | position y | position z | quaternion x | quaternion y | quaternion z | quaternion w |
| ---- | ---------- | ---------- | ---------- | ------------ | ------------ | ------------ | ------------ |


Columns are expected to be separated by commas by default.

It is recommended to provide a header at the beginning of the trajectory file. Header entries always begin with a "#".
Below you can find a table of all allowed header entries and their meaning.

| Header             | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #name              | The name provided here is displayed in the table view and in plots of the trajectory                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| #epsg              | [EPSG Code](https://epsg.io/) of the datum of the input positions. Required, if geodetic datum transformations are desired. Default: 0, meaning local coordinates without any known geodetic datum                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| #fields            | Describes the columns of the ASCII trajectory file. Separated with commas. <table>  <thead>  <th>field name</th>  <th>Meaning</th>  </tr>  </thead>  <tbody>  <tr>  <td>t</td>  <td>time</td>  </tr>  <tr>  <td>l</td>  <td>arc lengths in meters</td>  </tr>  <tr>  <td>px</td>  <td>position x / lat (degrees only)</td>  </tr>  <tr>  <td>py</td>  <td>position y / lon (degrees only) </td>  </tr>  <tr>  <td>pz</td>  <td>position z</td>  </tr> <tr>  <td>qx</td>  <td>quaternion x</td>  </tr> <tr>  <td>qy</td>  <td>quaternion y</td>  </tr> <tr>  <td>qz</td>  <td>quaternion z</td>  </tr> <tr>  <td>qw</td>  <td>quaternion w</td>  </tr> </tr> <tr>  <td>ex</td>  <td>euler angle x</td>  </tr> </tr> <tr>  <td>ey</td>  <td>euler angle y</td>  </tr> </tr> <tr>  <td>ez</td>  <td>euler angle z</td>  </tr> </tr> <tr>  <td>vx</td>  <td>speed x</td>  </tr> </tr> <tr>  <td>vy</td>  <td>speed y</td>  </tr> </tr> <tr>  <td>vz</td>  <td>speed z</td>  </tr> </tr> </tbody>  </table> Example: "#fields t,px,py,pz" Note: The only column that is allowed to appear multiple times is the "t" column. |
| #delimiter         | Delimiter used to separate the columns within the file. Default: ","                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| #nframe            | Definition of the navigation-frame the orientations of the trajectory refer to. Choices: "enu": East North Up or "ned": North East Down. Default: "enu"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| #rot_unit          | Unit of the orientations. Choices: "deg": Degree, "rad": Radians. Default: "rad"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| #time_format       | Format of the timestamps / dates. Choices: "unix": Unix timestamps (since 01-01-1970), "datetime": Human readable date-times. Default: "unix"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| #time_offset       | Offset in seconds that is applied to the imported timestamps. Default: 0.0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| #datetime_format   | Format of the datetimes. Only relevant if "time_format" is "datetime". Default: "%Y-%m-%d %H:%M:%S.%f"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| #datetime_timezone | Time zone of the timestamps. During import, all timestamps are converted to UTC considering the input time zone. Choices: [Time zone](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones) or "GPS"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| #sorting           | Sorting of the input data. Choices: "chrono": Chronologically sorted data (usually the case), "spatial": Spatially sorted data, i.e. along the arc length. Default: "chrono"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |


## Keyboard Shortcuts (GUI version only)

| Key       | Action                                                            |
| --------- | ----------------------------------------------------------------- |
| Ctrl + C  | Copy selected entry                                               |
| E         | Export selected entry                                             |
| M         | Merge selected trajectories                                       |
| P         | View properties of selected entry / entries                       |
| R         | Set selected trajectory as reference                              |
| Shift + R | Unset selected trajectory as reference                            |
| S         | Open trajectory settings                                          |
| T         | Transform selected trajectories to a different coordinate system. |
| U         | Rename selected entry                                             |
| V         | Plot all selected trajectories                                    |
