# run
streamlit run web/app.py

# ffmpeg
ffmpeg -ss 0 -t 2 -i "/run/media/manu/windows/workspace/Pixelle-Video/output/20260703_153409_8942/frames/01_segment.mp4" \
-vf "zoompan=z='min(zoom+0.00047430830039525733, 1.12)':x='(iw-ow)*on/253':y='(1080-oh)/2':d=1:s=1920x1080:fps=30" \
-c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p \
-c:a copy -y "/home/manu/tmp/move.mp4"