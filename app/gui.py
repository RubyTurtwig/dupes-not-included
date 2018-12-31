import PySimpleGUI as sg 

from .screen import Screen

def run_gui():
    """ Runs the GUI and program. """

    layout = [
        [sg.Text('Choose your duplicant template file.')], 
        [sg.InputText(), sg.FileBrowse()], 
        [sg.Text('Duplicant number:'), sg.InputCombo(['1', '2', '3']), sg.Button('Run')]]

    window = sg.Window('Dupes Not Included').Layout(layout)

    event, values = window.Read()
    window.Close()

    filename, number = values

    s = Screen()
    s.run(int(number) - 1, filename)