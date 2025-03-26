"""Timing utilities for Excel Header Mapper."""
import time
import statistics
from functools import wraps
from contextlib import contextmanager
from .display import console
from .formatting import format_time_delta

class ProcessingStats:
    """Track processing statistics for files and sheets"""
    def __init__(self):
        self.start_time = time.time()
        self.file_times = {}
        self.sheet_times = {}
        self.field_counts = {}
        self.success_count = 0
        self.error_count = 0
        self.total_fields_extracted = 0
        
    def record_file_start(self, file_path):
        """Record the start time for a file"""
        self.file_times[file_path] = {"start": time.time(), "end": None, "duration": None}
        
    def record_file_end(self, file_path, success=True, fields_extracted=0):
        """Record the end time and status for a file"""
        if file_path in self.file_times:
            end_time = time.time()
            self.file_times[file_path]["end"] = end_time
            self.file_times[file_path]["duration"] = end_time - self.file_times[file_path]["start"]
            self.file_times[file_path]["success"] = success
            self.file_times[file_path]["fields_extracted"] = fields_extracted
            
            if success:
                self.success_count += 1
                self.total_fields_extracted += fields_extracted
            else:
                self.error_count += 1
    
    def record_sheet_time(self, file_path, sheet_name, duration, fields_extracted=0):
        """Record processing time for a sheet"""
        if file_path not in self.sheet_times:
            self.sheet_times[file_path] = {}
        
        self.sheet_times[file_path][sheet_name] = {
            "duration": duration,
            "fields_extracted": fields_extracted
        }
        
    def get_average_file_time(self):
        """Get the average processing time per file"""
        durations = [data["duration"] for data in self.file_times.values() 
                    if data["duration"] is not None]
        return statistics.mean(durations) if durations else 0
    
    def get_estimated_time_remaining(self, files_remaining):
        """Estimate remaining time based on average processing time"""
        avg_time = self.get_average_file_time()
        return avg_time * files_remaining
    
    def get_total_duration(self):
        """Get total processing duration so far"""
        return time.time() - self.start_time
    
    def get_summary(self):
        """Get a summary of processing statistics"""
        total_files = self.success_count + self.error_count
        success_rate = (self.success_count / total_files * 100) if total_files > 0 else 0
        
        return {
            "total_duration": self.get_total_duration(),
            "total_files": total_files,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "total_fields_extracted": self.total_fields_extracted,
            "avg_fields_per_file": (self.total_fields_extracted / self.success_count) 
                                   if self.success_count > 0 else 0
        }

def time_function(func):
    """Decorator to measure and print execution time of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        
        # Format duration nicely
        if duration < 1:
            formatted_time = f"{duration*1000:.2f} ms"
        elif duration < 60:
            formatted_time = f"{duration:.2f} seconds"
        else:
            minutes = int(duration // 60)
            seconds = duration % 60
            formatted_time = f"{minutes} min {seconds:.2f} sec"
            
        console.print(f"[dim]Function {func.__name__} completed in {formatted_time}[/dim]")
        return result
    return wrapper

@contextmanager
def timed_section(description):
    """Context manager to time a section of code without using Status (to avoid display conflicts)"""
    console.print(f"[bold blue]{description}...[/bold blue]")
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        
        # Format duration nicely
        if duration < 1:
            formatted_time = f"{duration*1000:.2f} ms"
        elif duration < 60:
            formatted_time = f"{duration:.2f} seconds"
        else:
            minutes = int(duration // 60)
            seconds = duration % 60
            formatted_time = f"{minutes} min {seconds:.2f} sec"
        
        console.print(f"[green]✓[/green] {description} completed in {formatted_time}")
