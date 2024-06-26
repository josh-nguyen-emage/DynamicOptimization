import time
import subprocess

# def RunSimulation(idx):
#     cwd = "C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\AtenaPool\\"+str(idx)
#     subprocess.run("start /wait cmd /c \"C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe\" "+cwd+"\\G7-Cyl-Trial-1.inp a.out a.msg a.err",
#                 cwd=cwd,
#                 stdout=subprocess.DEVNULL,
#                 shell=True,
#                 check=True)

def RunSimulation(idx):
    cwd = "C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\AtenaPool\\"+str(idx)
    command = ["C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe", cwd + "\\G7-Cyl-Trial-1.inp", "a.out", "a.msg", "a.err"]
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    process.wait()

def RunSimulation_timeCheck(idx):
    cwd = "C:\\Users\\ADMIN\\Documents\\2.Working-Thinh\\AtenaPool\\" + str(idx)
    command = ["C:\\Program Files (x86)\\CervenkaConsulting\\AtenaV5\\AtenaConsole.exe", cwd + "\\G7-Cyl-Trial-1.inp", "a.out", "a.msg", "a.err"]

    try:
        process = subprocess.Popen(command, cwd="H:\\02.Working-Thinh\\ATENA-WORKING", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)

        start_time = time.time()
        while True:
            if process.poll() is not None:
                return True
            if time.time() - start_time > 60:  # 900 seconds = 15 minutes
                process.terminate()
                return False
            time.sleep(1)

    except subprocess.CalledProcessError:
        return False
