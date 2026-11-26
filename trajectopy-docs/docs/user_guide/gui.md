# GUI

Start the GUI from the Start menu (Windows app) or with `trajectopy` (Python package, see
[Command Line](command_line.md)).

## A Typical Workflow

1. **Load trajectories**: drop files onto the trajectory table or press *Add*. On Windows, `.traj` files can also be
   opened by double-click (if chosen during installation), and files can be passed on the command line.
2. **Set the reference**: right-click the ground truth and choose *Set as Reference* (`R`).
3. **Compare**: select one or more other trajectories, right-click and choose
   *Compare with Reference → Absolute Error (ATE), with Alignment*. The result appears in the results table.
4. **Inspect**: select the result to see its statistics in the properties table and its deviations in the
   preview; press `V` (or *Plot*) for all plots.
5. **Save**: *File → Save* (`Ctrl+S`) keeps everything for later in a project file; *Export…* (`E`) saves single
   trajectories or results.

## Main Window

The left side lists the trajectories (name, EPSG code, reference) and the results (name, type). Below them,
the properties table shows all properties of the selected trajectories or results side by side; *Export…* saves
them as CSV. The preview on the right plots the selected trajectories, or the selected ATE results colored by
their deviations. The drop-down next to the *Preview* checkbox in the top right corner switches between the top
view (*2D*) and an orbitable *3D* view (drag to rotate, wheel to zoom, middle-drag to pan). On slow systems, the
preview can be turned off with the checkbox; the tables and the properties then use the whole width.

Trajectory files (`.traj` and other ASCII files, ROS1 `.bag` files) are loaded into the trajectory table,
`.result` files into the results table. Long operations run in the background; a progress window shows that
Trajectopy is busy.

**Groups**: trajectories computed from another one (aligned, smoothed, sorted, … copies) are grouped with it
automatically, so that the results of one measurement run stay together. Groups can be collapsed; selecting a group
selects all of its entries. *Group → New Group…* (`Ctrl+G`) in the context menu groups any trajectories or results,
*Rename Group…* (`Ctrl+Shift+F2`) names a group, *Ungroup* (`Ctrl+Shift+G`) takes entries out of their groups and *Move to Group* moves
them into another group. Entries can also be dragged onto a group (or an entry) to move them there, or between the
rows to take them out of their group. Groups are saved in the project.

## Menu Bar

| Menu | Entry | Description |
| --- | --- | --- |
| File | New (`Ctrl+N`) | Starts a new, empty project with default plot settings. |
| | Open… (`Ctrl+O`) | Opens a project file (`.tpy`); replaces the current project. |
| | Open Recent | The last ten projects; *Clear Recent Projects* empties the list. |
| | Save (`Ctrl+S`) | Saves the project to its file (asks for a file the first time). |
| | Save As… (`Ctrl+Shift+S`) | Saves the project to a new file. |
| | Import Session Folder… | Opens a session folder saved by earlier versions of Trajectopy as a new project. |
| | Exit (`Ctrl+Q`) | Closes Trajectopy. |
| View | Log (`Ctrl+L`) | Shows or hides the log panel at the bottom of the main window. |
| Plot Settings | | Opens the [plot settings](settings.md#plot-settings), which apply to all plots opened afterwards. |
| Help | Check for Updates… | Checks GitHub for a newer version. |
| | Check for Updates at Startup | Checks quietly at every start and shows a link in the status bar if a newer version is available (on by default). |
| | About | Version and license information. |
| Theme | Dark / Light | Color theme. It starts with the system setting and follows when the system setting changes. |

## Projects

Projects keep everything in one file (`.tpy`): all trajectories with their processing settings, the
reference, all results, the order of the tables and the plot settings. The title bar shows the name of the
project and a `*` while it has unsaved changes. Before these are lost (*New*, *Open*, closing Trajectopy),
Trajectopy offers to save them. A project file can also be opened by dropping it onto one of the tables, by
double-click on Windows (if chosen during installation) or on the command line. Project files are binary, so
that large projects are saved and opened quickly; use *Export…* for files that other programs can read. Projects
saved by Trajectopy 5.1 can still be opened.

Closing Trajectopy offers to save unsaved changes and asks for confirmation while an operation is running.

## Log

Trajectopy writes what it does, warnings and errors to a log. *View → Log* (`Ctrl+L`) opens it at the
bottom of the main window; warnings are shown in orange, errors in red, and the context menu copies or clears the
messages. The log shows everything since Trajectopy was started, also if it is opened later, and it reopens at the
next start if it was open. The Windows app has no console window, so the log is the place to look when something
does not work as expected (Python installations also print the messages to the terminal they were started from).
If the window stops responding for more than half a second, the log names the code that blocked it, which helps
with bug reports.

## Updates

The update check only contacts GitHub (it sends no data except the installed version in the request
header); it can be turned off in the Help menu. Python installations are updated with
`pip install --upgrade "trajectopy[gui]"`, the Windows app with the installer of the new version.

## File Dialogs

The file dialogs open in the folder that was used last for the same kind of file (trajectories, results,
projects, settings, …), remember their size and list the usual and the recently used folders in their sidebar.
On Windows, Trajectopy uses its own file dialog instead of the one of the operating system: on some systems,
the Windows dialog takes several seconds to close after a file was chosen.

## Processing Settings of a Trajectory

Every trajectory has its own [processing settings](settings.md#processing-settings) (alignment, matching,
relative comparison, approximation, sorting), which are used when it is processed. Edit them with *Settings…*
(`S`) in its context menu; the editor can also import and export them as JSON. When a trajectory file is
loaded, a JSON file with the same name in the same folder (e.g. `run1.json` next to `run1.traj`) is used as its
settings.

## Context Menus

All actions are in the context menus of the selected trajectories or results (right-click). Hover over an entry
to see what it does. Entries that are not available for the current selection are greyed out and their hint
explains why, e.g. *Compare with Reference* needs a reference: right-click the reference trajectory and choose
*Set as Reference*.

The trajectory context menu is grouped by task:

| Group | Entries |
| --- | --- |
| View | *Plot*, *Playback in 3D* (body frame axes at the current pose), *Timeline* (time spans of the trajectories, e.g. to check their overlap) |
| Reference | *Set as Reference* / *Unset Reference* |
| Compare with Reference | *Absolute Error (ATE), without Alignment*, *Absolute Error (ATE), with Alignment*, *Relative Pose Error (RPE)*, *Extract Matching Poses* |
| Align with Reference | *Estimate Alignment*, *Interactive Alignment…* (see below), *Match First Pose*, *Apply Existing Alignment…* |
| Coordinate System | *Transform to EPSG…*, *Adapt EPSG from Reference*, *Set EPSG Code (No Transformation)…* |
| Process | *Approximate*, *Resample to Time Grid…*, *Rearrange Axes / Angles…*, *Sort Spatially*, *Switch Order: Time ↔ Path*, *Divide into Laps*, *Merge*, *Average* |
| Entry | *Settings…*, *Rename*, *Duplicate*, *Export…*, *Export to Google Earth (KML)…*, *Group*, *Remove*, *Remove Group* |

Processing adds new trajectories (e.g. *Approximate* adds a smoothed copy) and comparisons add results; only the
coordinate system entries, *Rearrange Axes / Angles…* and *Switch Order: Time ↔ Path* change the trajectory itself.

The results context menu offers *Plot*, *Rename*, *Duplicate*, *Export…*, *Group*, *Remove* and, for ATE results,
*Export to Google Earth (KML)…*. *Export Report or Summary Table…* saves the selected ATE and RPE results as an
HTML report or as a summary table (Markdown, CSV or LaTeX) with one row per result. ATE results of spatially sorted trajectories can also be switched between
chronological and spatial order with *Switch Order: Time ↔ Path*, e.g. to plot the deviations over time. For alignments, *Choose Parameters to Apply…* selects which of the estimated
parameters are used when the alignment is applied to a trajectory (*Apply Existing Alignment…*).

## Keyboard Shortcuts

Keyboard shortcuts are shown next to the entries of the context menus. They are active when the corresponding
table has focus and do what the menu entry does for the selected rows.

| Shortcut | Trajectories table | Results table |
| --- | --- | --- |
| `V` | Plot | Plot |
| `R` / `Shift+R` | Set as reference / unset reference | |
| `T` | Transform to EPSG | |
| `M` | Merge (two or more trajectories) | |
| `S` | Settings | |
| `F2` (or `U`) | Rename | Rename |
| `Ctrl+D` (or `Ctrl+C`) | Duplicate | Duplicate |
| `E` | Export | Export |
| `Delete` | Remove | Remove |
| `Ctrl+G` / `Ctrl+Shift+G` | New group / ungroup | New group / ungroup |
| `Ctrl+Shift+F2` | Rename group | Rename group |

## Interactive Alignment

Set a reference, then choose *Align with Reference → Interactive Alignment…* in the context menu of a trajectory.
The window helps in two situations: when the automatic alignment fails (unrelated frames, a different scale, a
large time offset), and when you want to understand *what* is wrong with an alignment, e.g. to calibrate a
leverarm or a time offset. It works in three steps; every step can be undone (`Ctrl+Z`, redo with `Ctrl+Y` or `Ctrl+Shift+Z`).

**1 · Place roughly.** Bring the trajectory onto the reference, in space and in time.

- **Auto align** tries several rough placements (the current one, the first pose, a fit on poses at equal time,
  the centers, the time from the speed profiles), keeps the closest and refines it by least squares.
- **Drag** the trajectory in the view to move it; `Ctrl`+drag rotates and `Shift`+drag scales it about the pivot
  (+). Rotations turn about the viewing direction (top, side or front view). Double-click places the pivot.
  Keyboard: arrows move, `Q`/`E` rotate (`Shift`: 10× larger, `Ctrl`: 10× finer).
- **Pick point pairs**: click a point on the trajectory and the reference position it belongs to. One pair fixes
  the position, two add the heading (and the scale with *fit scale*), three or more the full rotation.
- **Match first pose** puts the first pose onto the reference's first pose.
- **Sync time** finds the time offset by comparing the speed profiles, which do not depend on the coordinate
  frame. You can also drag left/right in the speed plot (*Over time*).

**2 · Diagnose and correct.** Once the trajectories overlap, the *deviation at equal time* remains: the position of
the trajectory minus that of the reference at the same timestamp. It is split into *along-track* (direction of
travel of the reference), *cross-track* (to its left) and *vertical*. Every alignment parameter leaves its own
signature in it:

| Error | Signature | View |
| --- | --- | --- |
| Time shift | along-track deviation proportional to the speed, none at standstills | *Time shift* |
| Leverarm | constant in the body frame, also at standstills | *Shift and leverarm* |
| Shift | constant in the world frame: in the body frame it changes with the heading | *Shift and leverarm* |
| Heading | tangential deviation (about the pivot) proportional to the distance from it | *Rotation and scale* |
| Scale | radial deviation proportional to the distance from the pivot | *Rotation and scale* |
| Roll, pitch | vertical deviation growing along x or y | *Rotation and scale* |

The **diagnosis** lists the parameters that explain the deviation, most important first. It considers only the
parameters checked with *estimate* (under *Values and estimated parameters*, preselected from the trajectory
settings): *Explains* is the part of the deviation (RMS) that only this parameter accounts for compared to the
other estimated ones, *Correction* its error. If a parameter that is not estimated would reduce the deviation
considerably, the diagnosis points it out. A parameter is listed only if it explains clearly more than noise
would (taking into account that the deviation of trajectories is correlated from pose to pose) and if it
matters compared to the deviation that remains after correcting everything; so correcting removes rows but
does not bring up new ones made of noise. The corrections are determined for the estimated parameters jointly, so that effects that look alike are attributed correctly; for a vehicle, for example, a
leverarm along the direction of travel and a time shift both deviate along-track, but only the time shift grows
with the speed. Parameters that cannot be told apart with the data at hand are named (e.g. the vertical leverarm
and the vertical shift of a platform that is always level).

Selecting a row opens the view with the signature of the parameter. By default, a view removes the effects of all
other parameters (*Hide effects of other parameters*), so that its plot shows just this signature plus what no parameter
explains; uncheck it to see the actual deviation. Hovering over the data marks the poses of that time in the main
view, and the *deviation arrows* there show the pattern on the map (parallel: shift; along the path: time shift;
swirl: heading; away from the pivot: scale).

The plots keep their axes while you drag, so that nothing jumps; they adapt when the data leaves the visible range
or fills only a small part of it. After zooming or panning a plot, double-click it to let its axes follow the data
again. The ranking of the parameters is updated once you pause, so that the rows do not reorder under the mouse;
the values of the placement (*Values and estimated parameters*) are folded away until you open them. Messages
appear in the status bar at the bottom.

Corrections are made with *Correct* in the diagnosis, with *Correct all* (all estimated parameters together),
or with the buttons of the views,
e.g. *Correct shift and leverarm*, which uses only the standstills, where a time shift has no effect. Every
correction is estimated together with all other parameters, so the order in which they are applied does not
matter.

**3 · Least squares.** *Refine by least squares* runs the rigorous spatio-temporal least squares alignment of the
parameters checked with *estimate* (preselected from the trajectory settings), starting from the placement shown;
it runs in the background and can be cancelled. The values of all parameters can also be typed in.
**Apply** aligns the trajectory as shown and saves the alignment; if the placement comes unchanged from the least
squares alignment, its statistics (e.g. the covariance) are saved, too.
