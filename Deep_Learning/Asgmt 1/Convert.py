import ffmpeg
import os

os.environ["PATH"] += os.pathsep + r"C:\ProgramData\chocolatey\bin"  # ffmpeg path

# opus to wav conversion
input_dir = 'Deep_Learning/Asgmt 1/Data_People/Raw/SUBHANALLAH'  # Change this to your folder path
output_dir = 'Deep_Learning/Asgmt 1/Data_People/SUBHANALLAH_O'  # Change this to your output folder path

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all files in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".opus"):
        # Construct the full input file path
        input_file = os.path.join(input_dir, filename)
        
        # Construct the output file path
        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.wav")
        
        # Convert the file using ffmpeg
        try:
            print(f"Converting {filename}...")
            ffmpeg.input(input_file).output(output_file, ar=16000, ac=1, acodec='pcm_s16le').run()
            print(f"Converted {filename} to {output_file}")
        except ffmpeg.Error as e:
            print(f"Error converting {filename}: {e}")
    
print("ALL FILES CONVERTED !! 🚀")
