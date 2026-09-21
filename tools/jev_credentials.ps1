function Enable-AikimiJevCredential {
    $SecretFile = Join-Path $env:LOCALAPPDATA 'Aikimi\secrets\jev-api-key.dpapi'
    $Ptr = [IntPtr]::Zero
    $Secure = $null
    try {
        if ($env:TYPESAFE_API_KEY) {
            $env:AIKIMI_JEV_ALLOW_CLOUD = '1'
            return
        }
        if (Test-Path -LiteralPath $SecretFile) {
            $Secure = Get-Content -LiteralPath $SecretFile -Raw | ConvertTo-SecureString
        } else {
            $Secure = Read-Host 'TYPESAFE_API_KEY (encrypted for this Windows user, outside Git)' -AsSecureString
            if ($Secure.Length -eq 0) { throw 'No Jev API key supplied.' }
            [void][System.IO.Directory]::CreateDirectory((Split-Path -Parent $SecretFile))
            [System.IO.File]::WriteAllText($SecretFile, (ConvertFrom-SecureString $Secure))
            $Acl = [System.Security.AccessControl.FileSecurity]::new()
            $Acl.SetAccessRuleProtection($true, $false)
            $UserSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
            $Acl.SetOwner($UserSid)
            $Acl.AddAccessRule([System.Security.AccessControl.FileSystemAccessRule]::new($UserSid, 'FullControl', 'Allow'))
            Set-Acl -LiteralPath $SecretFile -AclObject $Acl
        }
        $Ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
        $env:TYPESAFE_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($Ptr)
        $env:AIKIMI_JEV_ALLOW_CLOUD = '1'
    } finally {
        if ($Ptr -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($Ptr) }
        if ($Secure) { $Secure.Dispose() }
    }
}
