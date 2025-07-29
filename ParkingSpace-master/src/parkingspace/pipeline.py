# src/parkingspace/pipeline.py

import cv2
import numpy as np
import random
import time
import torch

def visualize_processing_steps(gray_frame, blur_frame, adaptive_thresh, median_thresh, 
                             prob_map_combined, enhanced_combined, final_combined, 
                             binary_map, dilated_image, show_debug=False):
    """
    Utility function to visualize intermediate processing steps.
    Set show_debug=True to display all intermediate images for debugging.
    """
    if show_debug:
        cv2.imshow("1. Gray Frame", gray_frame)
        cv2.imshow("2. Gaussian Blurred", blur_frame)
        cv2.imshow("3. Adaptive Threshold", adaptive_thresh)
        cv2.imshow("4. Median Filtered", median_thresh)
        cv2.imshow("5. Probability Map Combined", prob_map_combined)
        cv2.imshow("6. Enhanced Combined", enhanced_combined)
        cv2.imshow("7. Final Combined", final_combined)
        cv2.imshow("8. Binary Map", binary_map)
        cv2.imshow("9. Final Dilated", dilated_image)

def get_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None

def verify_nearby_vehicle(
    contour, vehicle_mask, aspect_ratio,
    search_radius=50, aspect_ratio_tolerance=0.4
):
    # Returns list of bounding boxes for similar vehicles near the given contour
    x, y, w, h = cv2.boundingRect(contour)
    search_x1 = max(x - search_radius, 0)
    search_y1 = max(y - search_radius, 0)
    search_x2 = min(x + w + search_radius, vehicle_mask.shape[1])
    search_y2 = min(y + h + search_radius, vehicle_mask.shape[0])

    roi_vehicle_mask = vehicle_mask[search_y1:search_y2, search_x1:search_x2]
    nearby_contours, _ = cv2.findContours(roi_vehicle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    matching_vehicles = []
    for nearby_contour in nearby_contours:
        nx, ny, nw, nh = cv2.boundingRect(nearby_contour)
        full_x = search_x1 + nx
        full_y = search_y1 + ny

        if nw != 0:
            nearby_aspect_ratio = nh / float(nw)
        else:
            nearby_aspect_ratio = 0

        if abs(nearby_aspect_ratio - aspect_ratio) <= aspect_ratio_tolerance:
            matching_vehicles.append((full_x, full_y, nw, nh))

    return matching_vehicles


def compute_parking_space_score(
    area, width, height, aspect_ratio, solidity,
    vehicle_bboxes, region_thresholds
):
    # Compute score for potential parking space
    score = 0.0

    ideal_area = (region_thresholds["min_area"] +
                  (region_thresholds["max_width"] * region_thresholds["max_height"])) / 2
    area_normalized = min(area / ideal_area, 1.0)
    area_score = 30 * area_normalized
    score += area_score

    max_asp = region_thresholds["max_aspect_ratio"]
    aspect_ratio_normalized = max_asp / aspect_ratio if aspect_ratio != 0 else 0
    aspect_ratio_normalized = min(aspect_ratio_normalized, 1.0)
    aspect_ratio_score = 20 * aspect_ratio_normalized
    score += aspect_ratio_score

    min_sol = region_thresholds["min_solidity"]
    if solidity > min_sol:
        solidity_normalized = min((solidity - min_sol) / (1.0 - min_sol), 1)
    else:
        solidity_normalized = 0
    solidity_score = 20 * solidity_normalized
    score += solidity_score

    # Bonus if at least 1 vehicle is near
    if vehicle_bboxes:
        score += 30

    return score


def process_frame(
    frame, vehicle_mask, prob_map_path,
    thresholds, upper_level_l, upper_level_m, upper_level_r,
    close_perp, far_side, close_side, far_perp, small_park,
    ignore_regions, show_debug=False
):
    # 1) Load probability map
    prob_map = cv2.imread(prob_map_path, cv2.IMREAD_GRAYSCALE)
    if prob_map is None:
        raise Exception(f"Failed to load probability map: {prob_map_path}")

    # Resize if needed
    if prob_map.shape != frame.shape[:2]:
        prob_map = cv2.resize(prob_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 2) Enhanced image preprocessing with Gaussian blur and adaptive thresholding
    # Convert frame to grayscale for additional processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    blur_frame = cv2.GaussianBlur(gray_frame, ksize=(3, 3), sigmaX=1)
    
    # Apply adaptive thresholding for better edge detection
    adaptive_thresh = cv2.adaptiveThreshold(
        blur_frame,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=25,
        C=16
    )
    
    # Apply median blur to further reduce noise
    median_thresh = cv2.medianBlur(adaptive_thresh, ksize=5)
    
    # 3) Invert vehicle mask
    vehicle_mask_scaled = (vehicle_mask * 255).astype(np.uint8)
    mask_img_inv = cv2.bitwise_not(vehicle_mask_scaled)

    # 4) Combine probability map with enhanced frame processing
    prob_map_combined = cv2.bitwise_and(prob_map, prob_map, mask=mask_img_inv)
    
    # Combine adaptive thresholding results with probability map for enhanced detection
    enhanced_combined = cv2.bitwise_and(median_thresh, median_thresh, mask=mask_img_inv)
    
    # Merge both approaches: probability map + adaptive thresholding
    final_combined = cv2.addWeighted(prob_map_combined, 0.6, enhanced_combined, 0.4, 0)

    # 5) Binarize & morphological ops on the enhanced combined image
    _, binary_map = cv2.threshold(final_combined, 50, 255, cv2.THRESH_BINARY)
    
    # Additional morphological operations for better contour detection
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(binary_map, kernel, iterations=6)
    
    # Apply dilation to recover some lost details (similar to Code/main.py)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

    # Optional: Visualize intermediate processing steps for debugging
    visualize_processing_steps(
        gray_frame, blur_frame, adaptive_thresh, median_thresh,
        prob_map_combined, enhanced_combined, final_combined,
        binary_map, dilated_image, show_debug
    )

    # 6) Find potential contours
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_contours_info = []
    for contour in contours:
        center = get_contour_center(contour)
        if center is None:
            continue

        # skip ignore
        if any(cv2.pointPolygonTest(region, center, False) >= 0 for region in ignore_regions):
            continue

        # basic geometry
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull) if hull is not None else 1
        solidity = area / hull_area if hull_area > 0 else 0

        # figure out which region we are in
        region_thresholds = None
        if cv2.pointPolygonTest(upper_level_l, center, False) >= 0:
            region_thresholds = thresholds['upper_level_l']
        elif cv2.pointPolygonTest(upper_level_m, center, False) >= 0:
            region_thresholds = thresholds['upper_level_m']
        elif cv2.pointPolygonTest(upper_level_r, center, False) >= 0:
            region_thresholds = thresholds['upper_level_r']
        elif cv2.pointPolygonTest(close_perp, center, False) >= 0:
            region_thresholds = thresholds['close_perp']
        elif cv2.pointPolygonTest(far_side, center, False) >= 0:
            region_thresholds = thresholds['far_side']
        elif cv2.pointPolygonTest(close_side, center, False) >= 0:
            region_thresholds = thresholds['close_side']
        elif cv2.pointPolygonTest(far_perp, center, False) >= 0:
            region_thresholds = thresholds['far_perp']
        elif cv2.pointPolygonTest(small_park, center, False) >= 0:
            region_thresholds = thresholds['small_park']
        else:
            continue

        if area < region_thresholds['min_area']:
            continue
        if solidity < region_thresholds['min_solidity']:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if not (region_thresholds['min_width'] <= w <= region_thresholds['max_width'] and
                region_thresholds['min_height'] <= h <= region_thresholds['max_height']):
            continue

        aspect_ratio = max(w, h) / float(min(w, h) if min(w, h) else 1)
        if aspect_ratio > region_thresholds['max_aspect_ratio']:
            continue

        # vehicles near this contour
        vehicle_bboxes = verify_nearby_vehicle(
            contour,
            vehicle_mask_scaled,
            aspect_ratio,
            search_radius=50,
            aspect_ratio_tolerance=0.4
        )

        # compute final score
        score = compute_parking_space_score(
            area=area,
            width=w,
            height=h,
            aspect_ratio=aspect_ratio,
            solidity=solidity,
            vehicle_bboxes=vehicle_bboxes,
            region_thresholds=region_thresholds
        )

        final_contours_info.append((contour, vehicle_bboxes, score, region_thresholds))

    # 8) Draw results
    labeled_image_bgr = frame.copy()
    total_spaces = 0
    avg_width_space = 200

    for i, (contour, vehicle_bboxes, score, region_thresholds) in enumerate(final_contours_info):
        if score >= 60:
            color = (0, 255, 0)
        elif score >= 30:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        x, y, w, h = cv2.boundingRect(contour)

        if w > avg_width_space:
            num_spaces = int(w / avg_width_space)
            space_width = w / num_spaces
            for j in range(num_spaces):
                sx = int(x + j * space_width)
                cv2.rectangle(labeled_image_bgr, (sx, y), (sx + int(space_width), y + h), color, 2)
                label = f"ID:{total_spaces + 1} Score:{int(score)}"
                cv2.putText(labeled_image_bgr, label,
                            (sx + int(space_width)//2, y + h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                total_spaces += 1
        else:
            cv2.rectangle(labeled_image_bgr, (x, y), (x + w, y + h), color, 2)
            label = f"ID:{total_spaces + 1} Score:{int(score)}"
            cv2.putText(labeled_image_bgr, label, (x + w//2, y + h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            total_spaces += 1

        # highlight vehicles in same color
        for (vx, vy, vw, vh) in vehicle_bboxes:
            cv2.rectangle(labeled_image_bgr, (vx, vy), (vx + vw, vy + vh), color, 2)

    return labeled_image_bgr, total_spaces
