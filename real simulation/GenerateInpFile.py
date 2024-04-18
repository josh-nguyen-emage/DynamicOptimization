import cv2
import random

def changeInpFile(E, S):
    stdInpFile = 'stdFile\G7-Cyl-Trial-1.inp'
    outputFile = 'real simulation\generatedInp'
    with open(stdInpFile, 'r') as file:
        lines = file.readlines()

    newFileName = "G7-Cyl_AutoGen_E"+str(E)+"_S"+str(S)
    lines[20] = "TASK name \""+newFileName+"\" \n"
    lines[22] = "Title \""+newFileName+"\" \n"

    for i in range(4265,4386,1):
        lines[i] = "SUPPORT SIMPLE node "+str(i-2486)+" dof 2 /**  GLOBAL **/ VALUE         "+str(-0.0003*(E/10))+"\n"

    del lines[4416:]

    for i in range(S):
        lines.append("STEP ID  "+str(i+1)+" STATIC NAME \"Load 1 BC#1\"\n")
        lines.append("LOAD CASE\n")
        lines.append("              1 *     "+str(1/S)+"\n")
        lines.append("Execute\n")
        if i%2 == 0:
            lines.append("STORE TO\n")
            resultype = "0"*4 + str(i+1)
            lines.append("\""+newFileName+"."+resultype+"\"\n")
        lines.append("\n\n")
    lines.append("\nDelete Load CASE  1 ;\n\n")
    lines.append("OUTPUT LOCATION OUTPUT_DATA DATA LIST \"load_|310|_REACTIONS #000010\" End ;\n")
    lines.append("OUTPUT LOCATION OUTPUT_DATA DATA LIST \"Deflection_|351|_DISPLACEMENTS #000010\" End ;")

    # Write the modified lines back to the file
    with open(outputFile+"\\"+newFileName+".inp", 'w') as file:
        file.writelines(lines)

changeInpFile(33, 100)
