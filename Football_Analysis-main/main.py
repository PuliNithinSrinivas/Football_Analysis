from utils import read_video, save_video, measure_distance
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from utils.bbox_utils import get_center_of_bbox
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


class BallKickDetector:
    def __init__(self, kick_speed_threshold=3.0):
        self.kick_speed_threshold = kick_speed_threshold  # Speed change threshold for a kick

    def detect_kicks(self, tracks):
        kicks = {}  # Store frame-wise kicks
        previous_ball_position = None
        previous_ball_speed = 0

        for frame_num, ball_data in enumerate(tracks['ball']):
            if len(ball_data) == 0:
                continue  # Skip frames where ball is missing

            ball_bbox = ball_data[1]['bbox']
            ball_position = get_center_of_bbox(ball_bbox)

            if previous_ball_position is not None:
                ball_speed = measure_distance(previous_ball_position, ball_position)

                if (ball_speed - previous_ball_speed) > self.kick_speed_threshold:
                    # A significant ball speed change detected - possible kick
                    kicker_id = self.get_kicking_player(tracks['players'][frame_num], ball_position)
                    if kicker_id is not None:
                        kicks[frame_num] = kicker_id

                previous_ball_speed = ball_speed

            previous_ball_position = ball_position

        return kicks

    def get_kicking_player(self, player_tracks, ball_position):
        min_distance = float('inf')
        kicker_id = None

        for player_id, player in player_tracks.items():
            player_bbox = player['bbox']
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < min_distance:
                min_distance = distance
                kicker_id = player_id

        return kicker_id if min_distance < 50 else None
    

def get_fastest_player(tracks):
    """
    Identify the player with the highest speed recorded across all frames.
    Also track total distance covered by each player.
    """
    max_speed = 0
    fastest_player_id = None
    fastest_player_team = None
    total_distance = {}

    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, track in player_tracks.items():
            speed = track.get('speed', 0)
            distance = track.get('distance', 0)
            team = track.get('team', 'Unknown')

            if player_id not in total_distance:
                total_distance[player_id] = 0
            total_distance[player_id] += distance

            if speed > max_speed:
                max_speed = speed
                fastest_player_id = player_id
                fastest_player_team = team
    
    return {
        'player_id': fastest_player_id,
        'speed': max_speed,
        'team': fastest_player_team,
        'total_distance': total_distance.get(fastest_player_id, 0)
    }

def get_potential_goal_scorer(tracks, goal_position):
    """
    Identify the player closest to the goal from each team who is most likely to score.
    Factors: Ball possession + Proximity to goal.
    Returns a dictionary with potential scorers for each team.
    """
    potential_scorers = {}
    min_distance_by_team = {}

    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, track in player_tracks.items():
            if track.get('has_ball', False):  # Only consider players with ball possession
                player_position = track.get('position_transformed', None)
                team = track.get('team', 'Unknown')

                if player_position:
                    distance_to_goal = measure_distance(player_position, goal_position)
                    
                    # Initialize team entry if not present
                    if team not in min_distance_by_team:
                        min_distance_by_team[team] = float('inf')
                        potential_scorers[team] = None

                    # Update if this player is closer to goal than previous best for their team
                    if distance_to_goal < min_distance_by_team[team]:
                        min_distance_by_team[team] = distance_to_goal
                        potential_scorers[team] = {
                            'player_id': player_id,
                            'distance_to_goal': distance_to_goal,
                            'team': team
                        }

    return potential_scorers

def main():
    # Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')
    
    # Initialize Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimation
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # Handle the case for the first frame or when no previous control exists
            if frame_num == 0 or not team_ball_control:
                team_ball_control.append('Unknown')
            else:
                team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)
    ball_kick_detector = BallKickDetector()
    kicks = ball_kick_detector.detect_kicks(tracks)

        # Print Ball Kicks
    print("\nBall Kicks Detected:")
    for frame, kicker in kicks.items():
        print(f"Frame {frame}: Player {kicker} kicked the ball.")

    # Identify the fastest player
    fastest_player = get_fastest_player(tracks)

    # Define goal position (assumed coordinates)
    goal_position = (50, 0)  # Adjust based on actual field mapping

    # Identify potential goal scorers for each team
    potential_scorers = get_potential_goal_scorer(tracks, goal_position)

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

    # Print key results
    print("\nFastest player from both Teams:")
    print(f"Fastest Player - ID: {fastest_player['player_id']}, Speed: {fastest_player['speed']:.2f} km/h, "
          f"Team: {fastest_player['team']}, Total Distance: {fastest_player['total_distance']:.2f} m")

    print("\nPotential Goal Scorers by Each Team:")
    for team, scorer in potential_scorers.items():
        if scorer and scorer['player_id'] is not None:  # Check if a scorer was found
            print(f"Team {team} - ID: {scorer['player_id']}, "
                  f"Distance to Goal: {scorer['distance_to_goal']:.2f} meters")
        else:
            print(f"Team {team} - No potential scorer identified (no ball possession)")

        

if __name__ == '__main__':
    main()





