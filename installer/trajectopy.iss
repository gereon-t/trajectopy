; Windows installer for Trajectopy (Inno Setup 6, https://jrsoftware.org/isinfo.php).
;
; Packages the one-folder PyInstaller build. Build it first, then compile this script:
;   uv run pyinstaller --noconfirm trajectopy.spec
;   iscc /DAppVersion=5.0.1 installer\trajectopy.iss
; The installer is written to dist\Trajectopy-<version>-setup.exe.
;
; It installs for the current user without administrator rights (or for all users, if chosen in the
; dialog), adds a Start menu entry and, optionally, a desktop icon and the .traj file association.
; Installing a newer version over an existing one updates it in place.

#define AppName "Trajectopy"
#define AppExe "Trajectopy.exe"
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
; never change the AppId: Windows identifies the installed application (and its updates) by it
AppId={{D3DF14CF-47F5-45DB-9C26-6F16E0A3170D}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Gereon Tombrink
AppPublisherURL=https://github.com/gereon-t/trajectopy
AppSupportURL=https://github.com/gereon-t/trajectopy/issues
AppUpdatesURL=https://github.com/gereon-t/trajectopy/releases
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename={#AppName}-{#AppVersion}-setup
SetupIconFile=..\trajectopy\gui\resources\icon-bg.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
LicenseFile=..\LICENSE
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ChangesAssociations=yes
; close a running Trajectopy before updating it
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "associate"; Description: "Open .traj files with {#AppName}"; GroupDescription: "File associations:"

[InstallDelete]
; an update replaces the bundled libraries completely, so that no files of the previous version remain
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\dist\Trajectopy\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
; .traj files: Trajectopy is the default program unless the user chose another one, and is always listed
; under "Open with". HKA is HKCU for a per-user and HKLM for an all-users installation.
Root: HKA; Subkey: "Software\Classes\.traj"; ValueType: string; ValueName: ""; ValueData: "Trajectopy.traj"; Flags: uninsdeletevalue; Tasks: associate
Root: HKA; Subkey: "Software\Classes\.traj\OpenWithProgids"; ValueType: string; ValueName: "Trajectopy.traj"; ValueData: ""; Flags: uninsdeletevalue; Tasks: associate
Root: HKA; Subkey: "Software\Classes\Trajectopy.traj"; ValueType: string; ValueName: ""; ValueData: "Trajectory File"; Flags: uninsdeletekey; Tasks: associate
Root: HKA; Subkey: "Software\Classes\Trajectopy.traj\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"",0"; Tasks: associate
Root: HKA; Subkey: "Software\Classes\Trajectopy.traj\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: associate
Root: HKA; Subkey: "Software\Classes\Applications\{#AppExe}\SupportedTypes"; ValueType: string; ValueName: ".traj"; ValueData: ""; Flags: uninsdeletekey; Tasks: associate

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
