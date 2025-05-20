#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import winshell
from win32com.client import Dispatch
import ctypes
from pathlib import Path

def is_admin():
    """Check if program is running with admin privileges"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def create_shortcut():
    """Create desktop shortcut with icon and hotkey"""
    try:
        # Get current working directory and desktop path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = os.path.dirname(script_dir)  # Parent directory of bat folder
        desktop = winshell.desktop()
        
        # Create shortcut object
        shell = Dispatch('WScript.Shell')
        
        # Create the debug mode shortcut
        shortcut_path = os.path.join(desktop, "Jarvis.lnk")
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = os.path.join(current_dir, "bat", "jarvis_window.bat")
        shortcut.WorkingDirectory = current_dir
        shortcut.IconLocation = os.path.join(current_dir, "pics", "ironman.ico")
        shortcut.Hotkey = "Ctrl+Alt+J"  # Set global hotkey Ctrl+Alt+J
        shortcut.Description = "Jarvis Voice Assistant (Debug Mode)"
        shortcut.save()
        
        print(f"Successfully created debug shortcut: {shortcut_path}")
        
        return True
    except Exception as e:
        print(f"Error creating shortcut: {str(e)}")
        return False

if __name__ == "__main__":
    if not is_admin():
        # Re-run with admin privileges
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
    else:
        # Check if user wants to create ICO file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = os.path.dirname(script_dir)  # Parent directory of bat folder
        
        ico_file = Path(os.path.join(current_dir, "pics", "ironman.ico"))
        jpg_file = Path(os.path.join(current_dir, "pics", "ironman.jpg"))
        
        if not ico_file.exists() and jpg_file.exists():
            try:
                from PIL import Image
                print("Converting JPG to ICO file...")
                img = Image.open(jpg_file)
                img.save(ico_file)
                print(f"Created icon file: {ico_file}")
            except ImportError:
                print("PIL library not installed, cannot convert image format")
                print("Please install PIL: pip install pillow")
        
        # Create shortcuts
        if create_shortcut():
            print("快捷方式创建成功！现在您可以：")
            print("1. 使用桌面图标启动Jarvis（调试模式）")
            print("2. 随时使用热键Ctrl+Alt+J启动Jarvis")
        
        input("按任意键退出...") 