"""Channel names can be edited without changing channel identity or data."""

import csv
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_ads1299"))

from history_replay import read_history_csv


class ChannelRenameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        name = "main_uart_channel_rename_test_import"
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "python_ads1299" / "main_uart_v1.2.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        cls.MainWindow = module.MainWindow
        cls.module_name = name

    def test_rename_updates_labels_dialogs_and_old_csv_remains_readable(self):
        from alpha_model_dialog import AlphaModelDialog
        from motor_imagery_dialog import MotorImageryDialog

        window = self.MainWindow()
        try:
            original_names = tuple(window.channel_names)
            window.alpha_model_dialog = AlphaModelDialog(window.channel_names, window)
            window.motor_imagery_dialog = MotorImageryDialog(window.channel_names, window)
            window.AdsData.emg_data[7].append(42.0)
            window.is_history_mode = True
            window.impedance_labels[7].setText("AF7\n22 kΩ")
            with patch(f"{self.module_name}.QInputDialog.getText", return_value=(" AF7-new ", True)):
                window.edit_channel_name(7)

            self.assertEqual(window.channel_names[7], "AF7-new")
            self.assertEqual(window.channel_buttons[7].text(), "AF7-new")
            self.assertEqual(window.channel_name_labels[7].get_text(), "AF7-new")
            self.assertEqual(window.impedance_labels[7].text(), "AF7-new\n22 kΩ")
            self.assertEqual(window.spectrum_lines[7].get_label(), "AF7-new")
            self.assertEqual(window.spectrum_ax.get_legend().get_texts()[7].get_text(), "AF7-new")
            self.assertEqual(window.alpha_model_dialog.channel_checks[7].text(), "AF7-new")
            self.assertIn("AF7-new", window.motor_imagery_dialog.c3_combo.itemText(8))
            self.assertEqual(window.AdsData.emg_data[7], [42.0])

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "old_250sps_raw.csv"
                with path.open("w", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(original_names)
                    writer.writerow(range(8))
                data = read_history_csv(path, window.channel_names, window.channel_name_history)
                self.assertEqual([channel[0] for channel in data], list(range(8)))
        finally:
            window.close()

    def test_waveform_name_double_click_opens_editor_even_outside_axes(self):
        from matplotlib.backend_bases import MouseEvent

        window = self.MainWindow()
        try:
            window.show()
            self.app.processEvents()
            window.canvas.draw()
            bounds = window.channel_name_labels[7].get_window_extent(window.canvas.get_renderer())
            event = MouseEvent(
                "button_press_event", window.canvas,
                bounds.x0 + bounds.width / 2, bounds.y0 + bounds.height / 2,
                button=1, dblclick=True,
            )
            with patch(f"{self.module_name}.QInputDialog.getText", return_value=("Fp2", True)):
                window._on_plot_button_press(event)
            self.assertEqual(window.channel_names[7], "Fp2")
        finally:
            window.close()

    def test_other_channel_labels_do_not_open_editor(self):
        from matplotlib.backend_bases import MouseEvent
        from PyQt5.QtCore import Qt
        from PyQt5.QtTest import QTest

        window = self.MainWindow()
        try:
            window.show()
            self.app.processEvents()
            window.canvas.draw()
            bounds = window.channel_name_labels[7].get_window_extent(window.canvas.get_renderer())
            spectrum_event = MouseEvent(
                "button_press_event", window.spectrum_canvas,
                bounds.x0 + bounds.width / 2, bounds.y0 + bounds.height / 2,
                button=1, dblclick=True,
            )
            with patch(f"{self.module_name}.QInputDialog.getText") as prompt:
                QTest.mouseDClick(window.channel_buttons[7], Qt.LeftButton)
                QTest.mouseDClick(window.impedance_labels[7], Qt.LeftButton)
                window._on_plot_button_press(spectrum_event)
            prompt.assert_not_called()
            self.assertEqual(window.channel_names[7], "AF7")
        finally:
            window.close()

    def test_empty_or_duplicate_name_is_rejected(self):
        window = self.MainWindow()
        try:
            for value in ("  ", "f5"):
                with patch(f"{self.module_name}.QInputDialog.getText", return_value=(value, True)), patch(
                    f"{self.module_name}.QMessageBox.warning"
                ) as warning:
                    window.edit_channel_name(7)
                warning.assert_called_once()
                self.assertEqual(window.channel_names[7], "AF7")
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
