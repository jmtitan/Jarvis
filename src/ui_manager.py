import tkinter as tk
from tkinter import ttk, messagebox
import threading
import asyncio
import yaml
import os
from typing import Callable, Any, Dict, Optional


class SettingsUI:
    """Settings UI Manager Class"""
    
    def __init__(self, 
                 tts_engine,
                 config: dict, 
                 save_config_callback: Callable,
                 play_audio_callback: Callable,
                 setup_hotkeys_callback: Callable):
        """
        Initialize the settings UI
        
        Args:
            tts_engine: TTS engine instance
            config: Configuration dictionary
            save_config_callback: Callback function to save configuration
            play_audio_callback: Callback function to play audio
            setup_hotkeys_callback: Callback function to setup hotkeys
        """
        self.tts_engine = tts_engine
        self.config = config
        self.save_config_callback = save_config_callback
        self.play_audio_callback = play_audio_callback
        self.setup_hotkeys_callback = setup_hotkeys_callback
        
    def show_settings(self):
        """Display settings window"""
        # Create a new thread to run the settings window to avoid blocking the main thread
        threading.Thread(target=self._create_settings_window, daemon=True).start()
        
    def _create_settings_window(self):
        """Create settings window"""
        # Create main window
        root = tk.Tk()
        root.title("Jarvis Assistant Settings")
        root.geometry("500x400")
        root.resizable(True, True)
        
        # Create tabs
        tab_control = ttk.Notebook(root)
        
        # TTS settings tab
        tts_tab = ttk.Frame(tab_control)
        tab_control.add(tts_tab, text="Voice Settings")
        
        # Hotkey settings tab
        hotkey_tab = ttk.Frame(tab_control)
        tab_control.add(hotkey_tab, text="Hotkey Settings")
        
        tab_control.pack(expand=1, fill="both")
        
        # Add TTS settings
        self._create_tts_settings(tts_tab)
        
        # Add hotkey settings
        self._create_hotkey_settings(hotkey_tab)
        
        # Add save button
        save_button = ttk.Button(root, text="Save Settings", command=lambda: self._save_settings(root))
        save_button.pack(pady=10)
        
        # Run window
        root.mainloop()
        
    def _create_tts_settings(self, parent):
        """Create TTS settings tab content"""
        settings_frame = ttk.LabelFrame(parent, text="Voice Synthesis Settings")
        settings_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Get current TTS settings
        current_settings = self.tts_engine.get_current_settings()
        available_voices = self.tts_engine.get_available_voices()
        
        # Voice selection
        ttk.Label(settings_frame, text="Select Voice:").grid(column=0, row=0, padx=10, pady=5, sticky=tk.W)
        voice_combo = ttk.Combobox(settings_frame, width=40)
        voice_combo['values'] = list(available_voices.keys())
        voice_combo.current(list(available_voices.keys()).index(current_settings['voice']) if current_settings['voice'] in available_voices else 0)
        voice_combo.grid(column=1, row=0, padx=10, pady=5)
        
        # Rate adjustment
        ttk.Label(settings_frame, text="Speech Rate:").grid(column=0, row=1, padx=10, pady=5, sticky=tk.W)
        rate_var = tk.DoubleVar(value=current_settings['rate'])
        rate_scale = ttk.Scale(settings_frame, from_=0.1, to=2.0, length=200, variable=rate_var)
        rate_scale.grid(column=1, row=1, padx=10, pady=5)
        rate_label = ttk.Label(settings_frame, text=f"{current_settings['rate']:.1f}")
        rate_label.grid(column=2, row=1, padx=5, pady=5)
        
        # Update rate label display
        def update_rate_label(*args):
            rate_label.configure(text=f"{rate_var.get():.1f}")
        rate_var.trace_add("write", update_rate_label)
        
        # Volume adjustment
        ttk.Label(settings_frame, text="Volume:").grid(column=0, row=2, padx=10, pady=5, sticky=tk.W)
        volume_var = tk.DoubleVar(value=current_settings['volume'])
        volume_scale = ttk.Scale(settings_frame, from_=0.1, to=2.0, length=200, variable=volume_var)
        volume_scale.grid(column=1, row=2, padx=10, pady=5)
        volume_label = ttk.Label(settings_frame, text=f"{current_settings['volume']:.1f}")
        volume_label.grid(column=2, row=2, padx=5, pady=5)
        
        # Update volume label display
        def update_volume_label(*args):
            volume_label.configure(text=f"{volume_var.get():.1f}")
        volume_var.trace_add("write", update_volume_label)
        
        # Test button
        test_button = ttk.Button(settings_frame, text="Test Voice", 
                                command=lambda: threading.Thread(
                                    target=lambda: asyncio.run(
                                        self._test_voice(voice_combo.get(), rate_var.get(), volume_var.get())
                                    )
                                ).start())
        test_button.grid(column=1, row=3, padx=10, pady=10)
        
        # Save settings to object instance
        def save_tts_settings():
            try:
                self.tts_engine.set_voice(voice_combo.get())
                self.tts_engine.set_rate(rate_var.get())
                self.tts_engine.set_volume(volume_var.get())
                return True
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save TTS settings: {e}")
                return False
                
        # Add to parent window's save functionality
        parent.save_tts_settings = save_tts_settings
        
    def _create_hotkey_settings(self, parent):
        """Create hotkey settings tab content"""
        hotkey_frame = ttk.LabelFrame(parent, text="Hotkey Settings")
        hotkey_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Get current hotkey settings
        current_hotkeys = self.config['hotkeys']
        
        # Toggle listening hotkey
        ttk.Label(hotkey_frame, text="Toggle Listening:").grid(column=0, row=0, padx=10, pady=5, sticky=tk.W)
        toggle_var = tk.StringVar(value=current_hotkeys['toggle_listening'])
        toggle_entry = ttk.Entry(hotkey_frame, textvariable=toggle_var)
        toggle_entry.grid(column=1, row=0, padx=10, pady=5)
        toggle_button = ttk.Button(hotkey_frame, text="Record", command=lambda: self._record_hotkey(toggle_var))
        toggle_button.grid(column=2, row=0, padx=5, pady=5)
        
        # Switch voice hotkey
        ttk.Label(hotkey_frame, text="Switch Voice:").grid(column=0, row=1, padx=10, pady=5, sticky=tk.W)
        voice_var = tk.StringVar(value=current_hotkeys['switch_voice'])
        voice_entry = ttk.Entry(hotkey_frame, textvariable=voice_var)
        voice_entry.grid(column=1, row=1, padx=10, pady=5)
        voice_button = ttk.Button(hotkey_frame, text="Record", command=lambda: self._record_hotkey(voice_var))
        voice_button.grid(column=2, row=1, padx=5, pady=5)
        
        # Adjust speed hotkey
        ttk.Label(hotkey_frame, text="Adjust Speed:").grid(column=0, row=2, padx=10, pady=5, sticky=tk.W)
        speed_var = tk.StringVar(value=current_hotkeys['adjust_speed'])
        speed_entry = ttk.Entry(hotkey_frame, textvariable=speed_var)
        speed_entry.grid(column=1, row=2, padx=10, pady=5)
        speed_button = ttk.Button(hotkey_frame, text="Record", command=lambda: self._record_hotkey(speed_var))
        speed_button.grid(column=2, row=2, padx=5, pady=5)
        
        # Save hotkey settings to config
        def save_hotkey_settings():
            try:
                # Temporary storage, waiting to save to file
                self.config['hotkeys']['toggle_listening'] = toggle_var.get()
                self.config['hotkeys']['switch_voice'] = voice_var.get()
                self.config['hotkeys']['adjust_speed'] = speed_var.get()
                
                # Reset hotkeys
                import keyboard
                keyboard.clear_all_hotkeys()
                self.setup_hotkeys_callback()
                return True
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save hotkey settings: {e}")
                return False
                
        # Add to parent window's save functionality
        parent.save_hotkey_settings = save_hotkey_settings
        
    def _record_hotkey(self, var):
        """Record hotkey"""
        dialog = tk.Toplevel()
        dialog.title("Record Hotkey")
        dialog.geometry("300x100")
        dialog.resizable(False, False)
        dialog.grab_set()  # Modal dialog
        
        ttk.Label(dialog, text="Press keyboard shortcut combination...").pack(pady=10)
        result_label = ttk.Label(dialog, text="")
        result_label.pack(pady=5)
        
        recorded_keys = []
        
        def on_key_press(e):
            key = e.name
            if key not in recorded_keys and key != 'escape':
                recorded_keys.append(key)
                result_label.config(text="+".join(recorded_keys))
                
        def on_key_release(e):
            if e.name == 'escape':
                dialog.destroy()  # Press ESC to cancel
            elif len(recorded_keys) > 0:
                hotkey = "+".join(recorded_keys)
                var.set(hotkey)
                dialog.destroy()
                
        # Bind key events
        dialog.bind("<KeyPress>", on_key_press)
        dialog.bind("<KeyRelease>", on_key_release)
        
    async def _test_voice(self, voice, rate, volume):
        """Test voice effect"""
        # Save current settings
        old_voice = self.tts_engine.voice
        old_rate = self.tts_engine.rate
        old_volume = self.tts_engine.volume
        
        try:
            # Apply temporary settings
            self.tts_engine.set_voice(voice)
            self.tts_engine.set_rate(rate)
            self.tts_engine.set_volume(volume)
            
            # Synthesize test speech
            text = "This is a test voice used to test the voice synthesis effect."
            audio_path = await self.tts_engine.speak(text)
            
            if audio_path:
                # Play test audio
                await self.play_audio_callback(audio_path)
        finally:
            # Restore original settings
            self.tts_engine.set_voice(old_voice)
            self.tts_engine.set_rate(old_rate)
            self.tts_engine.set_volume(old_volume)
            
    def _save_settings(self, root):
        """Save all settings"""
        # Call each tab's save function
        tts_saved = root.nametowidget(root.nametowidget(root.winfo_children()[0]).winfo_children()[0]).save_tts_settings()
        hotkey_saved = root.nametowidget(root.nametowidget(root.winfo_children()[0]).winfo_children()[1]).save_hotkey_settings()
        
        if tts_saved and hotkey_saved:
            # Save updated configuration to file
            try:
                self.save_config_callback()
                messagebox.showinfo("Success", "Settings saved successfully")
                root.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration file: {e}")


class TrayIconUI:
    """System Tray Icon Manager Class"""
    
    def __init__(self,
                 show_status_callback: Callable,
                 show_settings_callback: Callable,
                 quit_callback: Callable):
        """
        Initialize system tray
        
        Args:
            show_status_callback: Callback function to show status
            show_settings_callback: Callback function to show settings
            quit_callback: Callback function to quit program
        """
        self.show_status_callback = show_status_callback
        self.show_settings_callback = show_settings_callback
        self.quit_callback = quit_callback
        self.icon = None
        
    def create_tray_icon(self, icon_path=None):
        """
        Create system tray icon
        
        Args:
            icon_path: Icon file path, use default icon if None
        """
        import pystray
        from PIL import Image
        
        # Set icon
        if icon_path and os.path.exists(icon_path):
            print(f"Found icon: {icon_path}")
            image = Image.open(icon_path)
        else:
            print(f"Icon file not found, using default icon")
            image = Image.new('RGB', (64, 64), color='blue')
            
        # Create menu
        menu = pystray.Menu(
            pystray.MenuItem('Status', self.show_status_callback),
            pystray.MenuItem('Settings', self.show_settings_callback),
            pystray.MenuItem('Exit', self.quit_callback)
        )
        
        # Create icon
        self.icon = pystray.Icon("jarvis", image, "Jarvis Assistant", menu)
        print("System tray icon created successfully")
        
    def run(self):
        """Run system tray"""
        if self.icon:
            print("Starting system tray...")
            self.icon.run()
        else:
            print("Error: System tray icon not created")
            
    def stop(self):
        """Stop system tray"""
        if self.icon:
            self.icon.stop() 