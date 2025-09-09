// Integrated LearnOpenGL-style renderer with Delaunay-based terrain mesh

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>

#include <thread>
#include <chrono>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "C:\\Users\\mohamad\\Downloads\\BasicOpenGL-main\\BasicOpenGL-main\\includes\\learnopengl\\filesystem.h"
#include "C:\\Users\\mohamad\\Downloads\\BasicOpenGL-main\\BasicOpenGL-main\\includes\\learnopengl\\shader_m.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <opencv2/features2d.hpp>  // ORB
#include <filesystem>
namespace fs = std::filesystem;
bool map1 = false;
bool map2 = true;

struct Correspondence { glm::vec3 object; cv::Point2f image; };
std::vector<Correspondence> correspondences;
std::vector<glm::vec3> pairedObjectPoints;    // aligned 3D list (optional)
std::vector<cv::Point2f> pairedImagePoints;   // aligned 2D list (optional)
const char* CORR_FILE = "correspondences2.csv";



glm::vec3 pendingLeft;      bool hasPendingLeft  = false;  // 3D
cv::Point2f pendingRight;   bool hasPendingRight = false;  // 2D (right viewport pixels)

void appendCorrespondence(const glm::vec3& object, const cv::Point2f& image) {
    if(map1){
    CORR_FILE = "correspondences1.csv";
    }
    correspondences.push_back({object, image});
    pairedObjectPoints.push_back(object);
    pairedImagePoints.push_back(image);
    std::ofstream ofs(CORR_FILE, std::ios::app);
    if (!ofs) { std::cerr << "ERROR: cannot open " << CORR_FILE << "\n"; return; }
    ofs << std::fixed << std::setprecision(6)
        << object.x << "," << object.y << "," << object.z << ","
        << image.x  << "," << image.y  << "\n";
    std::cout << "💾 Saved pair to " << CORR_FILE
        << " => 3D(" << object.x << ", " << object.y << ", " << object.z
        << ")  2D(" << image.x << ", " << image.y << ")\n";
}

// === 2D feature detection overlay for RIGHT viewport ===
// --- ORB tuning (more features) ---
static int   ORB_NFEATURES      = 2500;   // ↑  (try 8000–12000 if you want)
static float ORB_SCALE_FACTOR   = 1.1f;   // finer pyramid
static int   ORB_NLEVELS        = 12;     // more levels
static int   ORB_EDGE_THRESHOLD = 31;
static int   ORB_FIRST_LEVEL    = 0;
static int   ORB_WTA_K          = 2;
static auto  ORB_SCORE_TYPE     = cv::ORB::HARRIS_SCORE; // better corner quality
static int   ORB_PATCH_SIZE     = 31;
static int   ORB_FAST_THRESHOLD = 8;      // ↓ easier to trigger corners (was ~20)

cv::Ptr<cv::ORB> orb = cv::ORB::create(
    ORB_NFEATURES, ORB_SCALE_FACTOR, ORB_NLEVELS,
    ORB_EDGE_THRESHOLD, ORB_FIRST_LEVEL, ORB_WTA_K,
    ORB_SCORE_TYPE, ORB_PATCH_SIZE, ORB_FAST_THRESHOLD
);
bool requestDetectKeypoints = false;          // set when pressing 'B'
int  maxKpToShow = ORB_NFEATURES;                       // show strongest N
std::vector<glm::vec3> kpOverlay;             // (x,y,0) in RIGHT viewport pixels
unsigned int kpVAO = 0, kpVBO = 0;            // GL buffers for overlay points

bool CaptureRightViewportToBGR(cv::Mat &outBGR);
void DetectKeypointsFromRightViewport();
void DrawKeypointOverlay(Shader& shader);

bool showKeypointsOverlay = false;

// Store all KP coords from the most recent detection (RIGHT viewport)
std::vector<cv::KeyPoint> g_lastRightKps;
std::vector<cv::Point2f>  currentKps2D;

// Saved sets of 2D points (from your 3 images)
// Each CSV row: "u,v" with u in [0, SCR_WIDTH/2), v in [0, SCR_HEIGHT)
std::vector<std::vector<cv::Point2f>> savedKpSets;
bool savedKpLoaded = false;
// === Saved 2D/3D from correspondences.csv ===
std::vector<cv::Point2f>  savedImagePointsAll; // (u,v) from CSV
std::vector<glm::vec3>  savedObjectPointsAll; // (x,y,z) from CSV (not required for K, but handy)

// Matching / overlay
int  K_MATCH_COUNT = 100;               // set 11 if you want 11
bool requestMatchK = false;            // set by K key
bool showMatchOverlay = false;

std::vector<glm::vec3> matchSavedOverlay;  // red
std::vector<glm::vec3> matchNewOverlay;    // cyan
unsigned int matchVAO1=0, matchVBO1=0;
unsigned int matchVAO2=0, matchVBO2=0;

// Optional: keep the last matched sets for downstream use (e.g., PnP)
std::vector<glm::vec3> lastMatchedObject3D;  // saved 3D for the matched saved 2D
std::vector<cv::Point2f> lastMatchedSaved2D;   // saved 2D (u,v)
std::vector<cv::Point2f> lastMatchedCurrent2D; // newly detected 2D (u,v)

bool requestAutoPnP = false;     // run auto PnP after we match
bool clearPrevPairsOnAuto = true; // optional: clear old pairs before filling new

// === Group saved correspondences into snapshots of 11 ===
struct Snapshot2D3D {
    std::vector<cv::Point2f> img; // saved (u,v) pixels
    std::vector<cv::Point3f> obj; // saved (x,y,z)
};
constexpr int SNAPSHOT_SIZE = 150;  // was 11
std::vector<Snapshot2D3D> savedSnaps;   // snapshots in file order
int lastSelectedSnapshotIdx = -1;       // which snapshot we picked last time

bool autoPairLeftFromRight = true;   // click in RIGHT auto-creates the LEFT 3D pair
bool autoSavePickedPairs   = false;  // set true if you want to append to CSV on each right pick

int counter = 0;

// === Feature bank for 9 snapshots (each ~2500 ORB features) ===
constexpr int BANK_COUNT    = 22;
constexpr int BANK_FEATURES = 2500; // target per snap
std::vector<std::string> bankPaths = map1
    ? std::vector<std::string>{
        "res/snaps1/snap_0.png","res/snaps1/snap_1.png","res/snaps1/snap_2.png",
        "res/snaps1/snap_3.png","res/snaps1/snap_4.png","res/snaps1/snap_5.png",
        "res/snaps1/snap_6.png","res/snaps1/snap_7.png","res/snaps1/snap_8.png",
        "res/snaps1/snap_9.png","res/snaps1/snap_10.png","res/snaps1/snap_11.png",
        "res/snaps1/snap_12.png","res/snaps1/snap_13.png","res/snaps1/snap_14.png",
        "res/snaps1/snap_15.png","res/snaps1/snap_16.png","res/snaps1/snap_17.png",
        "res/snaps1/snap_18.png","res/snaps1/snap_19.png","res/snaps1/snap_20.png",
        "res/snaps1/snap_21.png"
    }
    : std::vector<std::string>{
        "res/snaps2/snap_0.png","res/snaps2/snap_1.png","res/snaps2/snap_2.png",
        "res/snaps2/snap_3.png","res/snaps2/snap_4.png","res/snaps2/snap_5.png",
        "res/snaps2/snap_6.png","res/snaps2/snap_7.png","res/snaps2/snap_8.png",
        "res/snaps2/snap_9.png","res/snaps2/snap_10.png","res/snaps2/snap_11.png",
        "res/snaps2/snap_12.png","res/snaps2/snap_13.png","res/snaps2/snap_14.png",
        "res/snaps2/snap_15.png","res/snaps2/snap_16.png","res/snaps2/snap_17.png",
        "res/snaps2/snap_18.png","res/snaps2/snap_19.png","res/snaps2/snap_20.png",
        "res/snaps2/snap_21.png"
    };


struct SnapshotFeat {
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;   // CV_8U ORB descriptors
    cv::Mat gray;   // grayscale image (for debugging / drawing if needed)
};
std::vector<SnapshotFeat> featBank;      // size BANK_COUNT

// Current frame features (right view)
std::vector<cv::KeyPoint> curKps;
cv::Mat curDesc;

// For the chosen snapshot:
int selectedSnapIdx = -1;
std::vector<cv::DMatch> inlierMatchesSnapToCur; // inlier matches (snapshot->current)
cv::Mat H_snap_to_cur; // 3x3 homography


bool LoadSavedKeypointsCSV(const std::vector<std::string>& files);
void MatchTopKAndPrint();
void DrawMatchesOverlay(Shader& shader);


// ============================= Data Types =============================
struct Vertex { float x,y,z; float r,g,b; };

struct FrameSnapshot {
    glm::vec3 camPos;
    glm::vec3 camFront;
    glm::mat4 modelMatrix;
};

struct ColoredPoint { glm::vec3 pos; glm::vec3 color; };

struct Arrow {
    glm::vec3 position;  // tip position
    glm::vec3 direction; // forward (cameraFront when captured)
    glm::vec3 up;        // cameraUp when captured
};

struct CameraPose { glm::vec3 position, front, up; };


// ============================= Globals =============================

// Window
const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 900;

// Camera
glm::vec3 cameraPos   = glm::vec3(300, -300, 500);
glm::vec3 cameraFront = glm::vec3(0, 0, -1);
glm::vec3 cameraUp    = glm::vec3(0, 1, 0);

float yaw   = -90.0f;
float pitch =  9.0f;
float fov   = 100.0f;

bool  firstMouse = true;
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

// Terrain
std::vector<Vertex> terrainVertices;
std::vector<unsigned int> terrainIndices;
cv::Mat heightmap;
unsigned int terrainVAO=0, terrainVBO=0, terrainEBO=0;

// Recording & playback
std::vector<FrameSnapshot> savedFrames;
int framePlaybackIndex = 0;
bool framePlaybackMode = false;

unsigned int pathVAO = 0, pathVBO = 0;
std::vector<glm::vec3> pathLineVertices;

// Picking (now GPU depth-based)
bool PickingMode = false;
std::vector<ColoredPoint> coloredPointsLeft;
std::vector<ColoredPoint> coloredPointsRight;

std::vector<glm::vec3> colorPaletteLeft = {
    {1,0,0},{0,1,0},{0,0,1},{1,1,0},{1,0,1},{0,1,1},{1,0.5f,0},{0.5f,0,1}
};
std::vector<glm::vec3> colorPaletteRight = colorPaletteLeft;
int colorIndexLeft = 0, colorIndexRight = 0;

std::vector<Arrow> arrows;               // restored arrows list
std::vector<Arrow> arrows2;               // restored arrows list
int screenshotCounter = 0;               // for right-viewport screenshots
bool mouseArrow = false;

// Pose switching
CameraPose originalPose, computedPose;
bool useComputedPose = false;
bool computed = false;

std::vector<cv::Point2f> image2DPointsRight;



// ============================= Decls =============================
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
bool RayIntersectsTriangle(
    const glm::vec3 &orig, const glm::vec3 &dir,
    const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2,
    float &t, glm::vec3 &hitPoint);


//?://///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsureSnapsDir()
{
    try {
        fs::create_directories("res/snaps"); // no-op if it already exists
        return true;
    } catch (const std::exception& e) {
        std::cerr << "EnsureSnapsDir: failed to create res/snaps : " << e.what() << "\n";
        return false;
    }
}

bool LoadFeatureBank()
{
    featBank.clear();
    featBank.resize(BANK_COUNT);

    for (int i=0; i<BANK_COUNT; ++i) {
        cv::Mat img = cv::imread(bankPaths[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) { std::cerr << "Failed bank image: " << bankPaths[i] << "\n"; return false; }

        // detect ORB
        std::vector<cv::KeyPoint> kps;
        cv::Mat desc;
        orb->detectAndCompute(img, cv::noArray(), kps, desc);

        if ((int)kps.size() > BANK_FEATURES)
            cv::KeyPointsFilter::retainBest(kps, BANK_FEATURES);

        featBank[i].kps  = std::move(kps);
        featBank[i].desc = std::move(desc);
        featBank[i].gray = img;
        std::cout << "Loaded bank["<<i<<"] features=" << featBank[i].kps.size() << "\n";
    }
    return true;
}

// Returns best snapshot index and fills inlierMatchesSnapToCur + H.
// Score = #inliers (higher is better).
int SelectClosestSnapshotByFeatures()
{
    if (featBank.size() != BANK_COUNT || curDesc.empty()) return -1;

    cv::BFMatcher matcher(cv::NORM_HAMMING, false);

    int bestIdx = -1; int bestInliers = -1;
    cv::Mat bestH; std::vector<cv::DMatch> bestInlierMatches;

    for (int i=0; i<BANK_COUNT; ++i) {
        if (featBank[i].desc.empty()) continue;

        // KNN match snapshot -> current
        std::vector<std::vector<cv::DMatch>> knn;
        matcher.knnMatch(featBank[i].desc, curDesc, knn, 2);

        // Lowe ratio
        std::vector<cv::DMatch> good;
        good.reserve(knn.size());
        for (auto& v : knn) {
            if (v.size() < 2) continue;
            if (v[0].distance < 0.75f * v[1].distance) good.push_back(v[0]);
        }
        if (good.size() < 8) continue;

        // RANSAC homography on 2D->2D
        std::vector<cv::Point2f> src, dst;
        src.reserve(good.size()); dst.reserve(good.size());
        for (auto& m : good) {
            src.push_back(featBank[i].kps[m.queryIdx].pt); // snapshot
            dst.push_back(curKps[m.trainIdx].pt);          // current
        }

        std::vector<uchar> inlierMask;
        cv::Mat H = cv::findHomography(src, dst, cv::RANSAC, 3.0, inlierMask);
        if (H.empty()) continue;

        int inliers = 0;
        std::vector<cv::DMatch> inlierMatches; inlierMatches.reserve(good.size());
        for (size_t k=0; k<good.size(); ++k) {
            if (inlierMask[k]) { inlierMatches.push_back(good[k]); ++inliers; }
        }

        if (inliers > bestInliers) {
            bestInliers = inliers;
            bestIdx = i;
            bestH = H.clone();
            bestInlierMatches.swap(inlierMatches);
        }
    }

    selectedSnapIdx = bestIdx;
    inlierMatchesSnapToCur = std::move(bestInlierMatches);
    H_snap_to_cur = bestH;
    std::cout << "Selected snapshot = " << bestIdx << " with inliers = " << bestInliers << "\n";
    return bestIdx;
}




// Returns true and fills outLeft3D if a saved 2D is close to (u,v)
bool pickSaved3DFromSnapshot(float u, float v, glm::vec3& outLeft3D, float* outDistPx = nullptr) {
    if (lastSelectedSnapshotIdx < 0 || lastSelectedSnapshotIdx >= (int)savedSnaps.size()) return false;
    const auto& S = savedSnaps[lastSelectedSnapshotIdx];
    if (S.img.empty()) return false;

    float bestD2 = std::numeric_limits<float>::max();
    int best = -1;
    for (int i=0; i<(int)S.img.size(); ++i) {
        float dx = S.img[i].x - u, dy = S.img[i].y - v;
        float d2 = dx*dx + dy*dy;
        if (d2 < bestD2) { bestD2 = d2; best = i; }
    }
    if (best < 0) return false;
    const auto& P = S.obj[best];
    outLeft3D = glm::vec3(P.x, P.y, P.z);
    if (outDistPx) *outDistPx = std::sqrt(bestD2);
    return true;
}

bool LoadCorrespondencesCSV(const std::string& file)
{
    savedImagePointsAll.clear();
    savedObjectPointsAll.clear();
    savedSnaps.clear();

    std::ifstream ifs(file);
    if (!ifs) {
        std::cerr << "WARN: cannot open " << file << "\n";
        return false;
    }

    std::string line;
    Snapshot2D3D cur;
    int count = 0;

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        double x=0,y=0,z=0,u=0,v=0; char c1=0,c2=0,c3=0,c4=0;

        // Expect: x,y,z,u,v (commas). Be tolerant to whitespace.
        bool ok = false;
        if ((ss >> x >> c1 >> y >> c2 >> z >> c3 >> u >> c4 >> v) &&
            (c1==',' && c2==',' && c3==',' && c4==',')) {
            ok = true;
        } else {
            ss.clear(); ss.str(line);
            if (ss >> x >> y >> z >> u >> v) ok = true;
        }
        if (!ok) continue;

        // Flat storage (backward compatibility; optional)
        savedObjectPointsAll.emplace_back((float)x,(float)y,(float)z);
        savedImagePointsAll.emplace_back((float)u,(float)v);

        // Snapshot assembly (11 per snapshot)
        cur.obj.emplace_back((float)x,(float)y,(float)z);
        cur.img.emplace_back((float)u,(float)v);
        ++count;

        if (count == SNAPSHOT_SIZE) {
            savedSnaps.push_back(std::move(cur));
            cur = Snapshot2D3D{};
            count = 0;
        }
    }
    // If file isn’t an exact multiple of 11, keep the tail as a snapshot
    if (!cur.obj.empty()) {
        savedSnaps.push_back(std::move(cur));
    }

    std::cout << "Loaded " << savedImagePointsAll.size()
              << " rows from " << file << " -> " << savedSnaps.size()
              << " snapshot(s) of up to 11 points each.\n";

    return !savedSnaps.empty();
}


bool LoadSavedKeypointsCSV(const std::vector<std::string>& files)
{
    savedKpSets.clear();
    for (const auto& f : files) {
        std::ifstream ifs(f);
        if (!ifs) { std::cerr << "WARN: cannot open " << f << "\n"; continue; }
        std::vector<cv::Point2f> pts;
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            double u=0, v=0; char comma=0;
            if ( (ss >> u >> comma >> v) && (comma==',' || comma=='\t') ) {
                pts.emplace_back((float)u, (float)v);
            } else {
                // try space separated
                ss.clear(); ss.str(line);
                if (ss >> u >> v) pts.emplace_back((float)u, (float)v);
            }
        }
        if (!pts.empty()) {
            savedKpSets.push_back(std::move(pts));
            std::cout << "Loaded " << savedKpSets.back().size()
                      << " points from " << f << "\n";
        }
    }
    return !savedKpSets.empty();
}

// Find top-K closest pairs by 2D pixel distance (RIGHT viewport pixels)
void MatchTopKAndPrint()
{
    showMatchOverlay = false;
    matchSavedOverlay.clear();
    matchNewOverlay.clear();

    lastMatchedObject3D.clear();
    lastMatchedSaved2D.clear();
    lastMatchedCurrent2D.clear();

    if (savedSnaps.empty()) {
        std::cout << "No snapshots loaded from correspondences.csv\n";
        return;
    }
    if (currentKps2D.empty()) {
        std::cout << "No current 2D features detected.\n";
        return;
    }

    // --- Helper: score a snapshot by mean distance of top-K nearest pairs (unique saved) ---
    auto scoreSnapshot = [&](const Snapshot2D3D& S, int K, double& outMean, std::vector<int>& outSavedIdx, std::vector<int>& outCurrIdx) -> bool {
        if (S.img.empty()) return false;

        struct Pair { float d2; int curIdx; int savIdx; };
        std::vector<Pair> pairs; pairs.reserve(currentKps2D.size());

        // For each current point, find nearest saved in this snapshot
        for (int ci=0; ci<(int)currentKps2D.size(); ++ci) {
            const auto& c = currentKps2D[ci];
            float bestD2 = std::numeric_limits<float>::max();
            int bestSi = -1;
            for (int si=0; si<(int)S.img.size(); ++si) {
                const auto& s = S.img[si];
                float dx = s.x - c.x, dy = s.y - c.y;
                float d2 = dx*dx + dy*dy;
                if (d2 < bestD2) { bestD2 = d2; bestSi = si; }
            }
            if (bestSi >= 0) pairs.push_back({bestD2, ci, bestSi});
        }
        if (pairs.empty()) return false;

        std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b){ return a.d2 < b.d2; });

        // pick unique saved points, up to K
        std::vector<char> usedSaved(S.img.size(), 0);
        outSavedIdx.clear(); outCurrIdx.clear();
        outSavedIdx.reserve(K); outCurrIdx.reserve(K);

        for (const auto& p : pairs) {
            if ((int)outSavedIdx.size() >= K) break;
            if (!usedSaved[p.savIdx]) {
                usedSaved[p.savIdx] = 1;
                outSavedIdx.push_back(p.savIdx);
                outCurrIdx.push_back(p.curIdx);
            }
        }
        if (outSavedIdx.empty()) return false;

        double sum = 0.0;
        for (size_t i=0; i<outSavedIdx.size(); ++i) {
            auto& s = S.img[outSavedIdx[i]];
            auto& c = currentKps2D[outCurrIdx[i]];
            float dx = s.x - c.x, dy = s.y - c.y;
            sum += std::sqrt(dx*dx + dy*dy);
        }
        outMean = sum / outSavedIdx.size();
        return true;
    };

    // --- 1) Pick best snapshot ---
    int bestIdx = -1;
    double bestScore = 1e18;
    std::vector<int> bestSavedIdx, bestCurrIdx;

    for (int si=0; si<(int)savedSnaps.size(); ++si) {
        double score = 0.0;
        std::vector<int> tmpSaved, tmpCurr;
        if (scoreSnapshot(savedSnaps[si], K_MATCH_COUNT, score, tmpSaved, tmpCurr)) {
            if (score < bestScore) {
                bestScore = score;
                bestIdx = si;
                bestSavedIdx = std::move(tmpSaved);
                bestCurrIdx  = std::move(tmpCurr);
            }
        }
    }

    if (bestIdx < 0) {
        std::cout << "No suitable snapshot found for matching.\n";
        return;
    }
    lastSelectedSnapshotIdx = bestIdx;
    const auto& Snap = savedSnaps[bestIdx];

    std::cout << "Selected snapshot #" << bestIdx
              << " (score=" << bestScore << ", used " << bestSavedIdx.size() << " pairs)\n";

    // --- 2) Print matches (2D saved, 2D new, distance, and saved 3D) ---
    std::cout << "== Top " << bestSavedIdx.size() << " pairs from snapshot #" << bestIdx << " ==\n";
    for (size_t i=0; i<bestSavedIdx.size(); ++i) {
        int si = bestSavedIdx[i];
        int ci = bestCurrIdx[i];
        const auto& s2 = Snap.img[si];
        const auto& s3 = Snap.obj[si];
        const auto& c2 = currentKps2D[ci];
        float dx = s2.x - c2.x, dy = s2.y - c2.y;
        float d = std::sqrt(dx*dx + dy*dy);
        std::cout << i << ") SAVED 2D=(" << s2.x << "," << s2.y << ")  <->  NEW 2D=("
                  << c2.x << "," << c2.y << ")  d=" << d << " px\n"
                  << "    SAVED 3D=(" << s3.x << "," << s3.y << "," << s3.z << ")\n";
    }

    // --- 3) Fill lastMatched* for downstream auto-PnP ---
    lastMatchedObject3D.clear();
    lastMatchedSaved2D.clear();
    lastMatchedCurrent2D.clear();
    for (size_t i=0; i<bestSavedIdx.size(); ++i) {
        int si = bestSavedIdx[i];
        int ci = bestCurrIdx[i];
        const auto& s3 = Snap.obj[si];
        const auto& s2 = Snap.img[si];
        const auto& c2 = currentKps2D[ci];
        lastMatchedObject3D.emplace_back(s3.x, s3.y, s3.z); // glm::vec3
        lastMatchedSaved2D.push_back(s2);
        lastMatchedCurrent2D.push_back(c2);
    }

    // --- 4) Build overlay (saved=RED, new=CYAN) ---
    matchSavedOverlay.clear();
    matchNewOverlay.clear();
    matchSavedOverlay.reserve(bestSavedIdx.size());
    matchNewOverlay.reserve(bestSavedIdx.size());
    for (size_t i=0; i<bestSavedIdx.size(); ++i) {
        int si = bestSavedIdx[i];
        int ci = bestCurrIdx[i];
        const auto& s2 = Snap.img[si];
        const auto& c2 = currentKps2D[ci];
        matchSavedOverlay.emplace_back(s2.x, s2.y, 0.0f);
        matchNewOverlay.emplace_back(c2.x, c2.y, 0.0f);
    }

    if (!matchVAO1) { glGenVertexArrays(1, &matchVAO1); glGenBuffers(1, &matchVBO1); }
    if (!matchVAO2) { glGenVertexArrays(1, &matchVAO2); glGenBuffers(1, &matchVBO2); }

    glBindVertexArray(matchVAO1);
    glBindBuffer(GL_ARRAY_BUFFER, matchVBO1);
    glBufferData(GL_ARRAY_BUFFER, matchSavedOverlay.size()*sizeof(glm::vec3),
                 matchSavedOverlay.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(matchVAO2);
    glBindBuffer(GL_ARRAY_BUFFER, matchVBO2);
    glBufferData(GL_ARRAY_BUFFER, matchNewOverlay.size()*sizeof(glm::vec3),
                 matchNewOverlay.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    showMatchOverlay = true;
}




void DrawMatchesOverlay(Shader& shader)
{
    if (!showMatchOverlay) return;

    // 2D ortho that maps RIGHT viewport pixels to NDC
    glm::mat4 proj2D = glm::ortho(0.0f, (float)(SCR_WIDTH/2), (float)SCR_HEIGHT, 0.0f, -1.0f, 1.0f);
    shader.use();
    shader.setMat4("projection", proj2D);
    shader.setMat4("view", glm::mat4(1.0f));
    shader.setMat4("model", glm::mat4(1.0f));
    glDisable(GL_DEPTH_TEST);

    // Saved points = RED
    if (matchVAO1 && !matchSavedOverlay.empty()) {
        shader.setBool("useUniformColor", true);
        shader.setVec3("color", glm::vec3(1.0f, 0.0f, 0.0f));
        glBindVertexArray(matchVAO1);
        glPointSize(10.0f);
        glDrawArrays(GL_POINTS, 0, (GLsizei)matchSavedOverlay.size());
    }

    // New points = CYAN
    if (matchVAO2 && !matchNewOverlay.empty()) {
        shader.setBool("useUniformColor", true);
        shader.setVec3("color", glm::vec3(0.0f, 1.0f, 1.0f));
        glBindVertexArray(matchVAO2);
        glPointSize(10.0f);
        glDrawArrays(GL_POINTS, 0, (GLsizei)matchNewOverlay.size());
    }

    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
}


bool CaptureRightViewportToBGR(cv::Mat &outBGR)
{
    int w = SCR_WIDTH / 2;
    int h = SCR_HEIGHT;
    std::vector<unsigned char> rgb(w * h * 3);

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glReadPixels(SCR_WIDTH/2, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, rgb.data());

    cv::Mat img(h, w, CV_8UC3, rgb.data());
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::flip(img, img, 0); // flip because GL origin is bottom-left

    outBGR = img.clone(); // deep copy
    return !outBGR.empty();
}

void DetectKeypointsFromRightViewport()
{
    // 1) Capture the right half of the window into a BGR image
    cv::Mat frameBGR;
    if (!CaptureRightViewportToBGR(frameBGR)) {
        std::cerr << "DetectKeypointsFromRightViewport: capture failed.\n";
        kpOverlay.clear();
        curKps.clear();
        curDesc.release();
        return;
    }

    // 2) Grayscale + (optional) CLAHE to boost local contrast
    cv::Mat gray;
    cv::cvtColor(frameBGR, gray, cv::COLOR_BGR2GRAY);
    static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8,8));
    clahe->apply(gray, gray);

    // 3) ORB detect+compute
    curKps.clear();
    curDesc.release();
    orb->detectAndCompute(gray, cv::noArray(), curKps, curDesc);

    // 4) Keep strongest BANK_FEATURES (~2500), then recompute descriptors to align rows
    if ((int)curKps.size() > BANK_FEATURES) {
        cv::KeyPointsFilter::retainBest(curKps, BANK_FEATURES);
        orb->compute(gray, curKps, curDesc);  // recompute so descriptors match retained keypoints
    }

    // 5) (Optional) simple spatial NMS to reduce clumping
    const float MIN_DIST = 4.0f; // px; tune 3–8
    if (!curKps.empty() && MIN_DIST > 0.0f) {
        std::vector<cv::KeyPoint> kept; kept.reserve(curKps.size());
        for (const auto& kp : curKps) {
            bool Far = true;
            for (const auto& q : kept) {
                float dx = kp.pt.x - q.pt.x, dy = kp.pt.y - q.pt.y;
                if (dx*dx + dy*dy < MIN_DIST*MIN_DIST) { Far = false; break; }
            }
            if (Far) kept.push_back(kp);
        }
        curKps.swap(kept);
        orb->compute(gray, curKps, curDesc);  // recompute for the NMS-filtered set
    }

    // 6) Build/update 2D overlay points (RIGHT viewport pixel coords)
    kpOverlay.clear();
    kpOverlay.reserve(curKps.size());
    for (const auto& kp : curKps) kpOverlay.emplace_back(kp.pt.x, kp.pt.y, 0.0f);

    if (kpVAO == 0) { glGenVertexArrays(1, &kpVAO); glGenBuffers(1, &kpVBO); }
    glBindVertexArray(kpVAO);
    glBindBuffer(GL_ARRAY_BUFFER, kpVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 kpOverlay.size() * sizeof(glm::vec3),
                 kpOverlay.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    // 7) (Optional) if you keep a plain 2D list elsewhere, populate it here:
    // currentKps2D.clear();
    // currentKps2D.reserve(curKps.size());
    // for (const auto& kp : curKps) currentKps2D.push_back(kp.pt);
}


void DrawKeypointOverlay(Shader& shader)
{
    if (kpOverlay.empty() || kpVAO == 0) return;

    shader.use();
    // 2D ortho that maps RIGHT viewport pixels to NDC
    glm::mat4 proj2D = glm::ortho(0.0f, (float)(SCR_WIDTH/2), (float)SCR_HEIGHT, 0.0f, -1.0f, 1.0f);
    shader.setMat4("projection", proj2D);
    shader.setMat4("view", glm::mat4(1.0f));
    shader.setMat4("model", glm::mat4(1.0f));
    shader.setBool("useUniformColor", true);
    shader.setVec3("color", glm::vec3(0.0f, 1.0f, 0.0f)); // green

    glDisable(GL_DEPTH_TEST);
    glBindVertexArray(kpVAO);
    glPointSize(6.0f);
    glDrawArrays(GL_POINTS, 0, (GLsizei)kpOverlay.size());
    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
}


void LoadUnstructuredTerrain(const std::string &path, int sampleCount = 5000)
{
    heightmap = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (heightmap.empty())
    {
        std::cerr << "Failed to load heightmap: " << path << std::endl;
        return;
    }

    terrainVertices.clear();
    terrainIndices.clear();

    std::vector<cv::Point2f> samplePoints;
    cv::RNG rng;
    for (int i = 0; i < sampleCount; ++i)
    {
        float x = rng.uniform(0.0f, (float)heightmap.cols);
        float y = rng.uniform(0.0f, (float)heightmap.rows);
        samplePoints.emplace_back(x, y);
    }

    cv::Subdiv2D subdiv(cv::Rect(0, 0, heightmap.cols, heightmap.rows));
    for (const auto &pt : samplePoints)
        subdiv.insert(pt);

    std::vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);

    std::map<std::pair<int, int>, int> vertexIndexMap;
    auto getIndex = [&](int x, int y) -> int
    {
        auto key = std::make_pair(x, y);
        auto it = vertexIndexMap.find(key);
        if (it != vertexIndexMap.end())
            return it->second;

        uchar gray = heightmap.at<uchar>(std::clamp(y, 0, heightmap.rows - 1), std::clamp(x, 0, heightmap.cols - 1));
        float z = (gray / 255.0f) * 80.0f;
        int idx = terrainVertices.size();
        float maxHeight = 50.0f;
        float t = std::clamp(z / maxHeight, 0.0f, 1.0f);
        float r = t;
        float g = 1.0f - fabs(t - 0.5f) * 2.0f;
        float b = 1.0f - t;
        terrainVertices.push_back({(float)x, (float)-y, z, r, g, b});
        vertexIndexMap[key] = idx;
        return idx;
    };

    for (const auto &t : triangleList)
    {
        int x1 = static_cast<int>(t[0]), y1 = static_cast<int>(t[1]);
        int x2 = static_cast<int>(t[2]), y2 = static_cast<int>(t[3]);
        int x3 = static_cast<int>(t[4]), y3 = static_cast<int>(t[5]);

        if (x1 < 0 || x1 >= heightmap.cols || y1 < 0 || y1 >= heightmap.rows)
            continue;
        if (x2 < 0 || x2 >= heightmap.cols || y2 < 0 || y2 >= heightmap.rows)
            continue;
        if (x3 < 0 || x3 >= heightmap.cols || y3 < 0 || y3 >= heightmap.rows)
            continue;

        int i1 = getIndex(x1, y1);
        int i2 = getIndex(x2, y2);
        int i3 = getIndex(x3, y3);

        terrainIndices.push_back(i1);
        terrainIndices.push_back(i2);
        terrainIndices.push_back(i3);
    }
}

bool isRecording = false;
bool isPlayback = false;
std::vector<glm::vec3> recordedPositions;
std::vector<glm::vec3> recordedFronts;
int playbackIndex = 0;

void computeCameraPose()
{

    std::cout << "coloredPointsLeft size: " << coloredPointsLeft.size() << std::endl;
    std::cout << "coloredPointsRight size: " << coloredPointsRight.size() << std::endl;
    // 1. Validate 2D-3D correspondences
        // 1) Validate 2D-3D correspondences (LEFT = 3D, RIGHT = 2D pixels)
    if (coloredPointsLeft.size() < 4 || image2DPointsRight.size() < 4) {
        std::cout << "Error: Need at least 4 3D-2D pairs.\n";
        return;
    }
    if (coloredPointsLeft.size() != image2DPointsRight.size()) {
        std::cout << "Error: Mismatched counts (3D left vs 2D right).\n";
        return;
    }

    // 2. Build 3D-2D correspondence vectors
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    for (size_t i = 0; i < coloredPointsLeft.size(); ++i)
    {
        objectPoints.emplace_back(coloredPointsLeft[i].pos.x, coloredPointsLeft[i].pos.y, coloredPointsLeft[i].pos.z);
        // imagePoints.emplace_back(coloredPointsRight[i].pos.x, coloredPointsRight[i].pos.y);
        imagePoints.emplace_back(image2DPointsRight[i]);
    }

    // 3. Camera intrinsic parameters
    float fovy = 65.0f; // degrees
    float height = 900.0f;
    float width = 600.0f;
    float fy = (height / 2.0f) / tan(glm::radians(fovy) / 2.0f);
    float fx = fy;
    float cx = width / 2.0f;
    float cy = height / 2.0f;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                            0, fy, cy,
                            0, 0, 1);
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F); // No distortion assumed

    // 4. Solve PnP
    cv::Mat rvec, tvec;
    bool success = cv::solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs,
                                      rvec, tvec, false, 100, 8.0, 0.99);
    if (!success)
    {
        std::cout << "solvePnPRansac failed!\n";
        return;
    }
    // 2. Refine pose with Levenberg-Marquardt
    cv::solvePnPRefineLM(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

    // 5. Convert rotation vector to rotation matrix
    cv::Mat rotMat;
    cv::Rodrigues(rvec, rotMat);

    // 6. Convert to OpenGL camera parameters
    cv::Mat R_inv = rotMat.t(); // R^-1 = Rᵗ for rotation matrix
    cv::Mat camPos_cv = -R_inv * tvec;

    glm::vec3 computedPos(
        static_cast<float>(camPos_cv.at<double>(0)),
        static_cast<float>(camPos_cv.at<double>(1)),
        static_cast<float>(camPos_cv.at<double>(2)));

    glm::vec3 front(
        static_cast<float>(rotMat.at<double>(2, 0)),
        static_cast<float>(rotMat.at<double>(2, 1)),
        static_cast<float>(rotMat.at<double>(2, 2)));

    glm::vec3 up(
        -static_cast<float>(rotMat.at<double>(1, 0)),
        -static_cast<float>(rotMat.at<double>(1, 1)),
        -static_cast<float>(rotMat.at<double>(1, 2)));

    // 7. Save original pose
    originalPose.position = cameraPos;
    originalPose.front = cameraFront;
    originalPose.up = cameraUp;

    // 8. Apply new computed pose
    computedPose.position = computedPos;
    computedPose.front = glm::normalize(front);
    computedPose.up = glm::normalize(up);

    cameraPos = computedPose.position;
    cameraFront = computedPose.front;
    cameraUp = computedPose.up;

    // 9. Debug print
    std::cout << "Computed Camera Pose:\n"
              << "Position: (" << cameraPos.x << ", " << cameraPos.y << ", " << cameraPos.z << ")\n"
              << "Front: (" << cameraFront.x << ", " << cameraFront.y << ", " << cameraFront.z << ")\n"
              << "Up: (" << cameraUp.x << ", " << cameraUp.y << ", " << cameraUp.z << ")\n";

    std::cout << "Original Camera Pose:\n"
              << "Position: (" << originalPose.position.x << ", " << originalPose.position.y << ", " << originalPose.position.z << ")\n"
              << "Front: (" << originalPose.front.x << ", " << originalPose.front.y << ", " << originalPose.front.z << ")\n"
              << "Up: (" << originalPose.up.x << ", " << originalPose.up.y << ", " << originalPose.up.z << ")\n";

    std::vector<cv::Point2f> reprojected;
    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, reprojected);

    double totalError = 0.0;
    for (size_t i = 0; i < imagePoints.size(); ++i)
    {
        double err = cv::norm(imagePoints[i] - reprojected[i]);
        std::cout << "Reprojection error [Point " << i << "]: " << err << " px\n";
        totalError += err;
    }
    std::cout << "Average reprojection error: " << (totalError / imagePoints.size()) << " px\n";
}
void BuildPairsFromSelectedSnapshotAndRunPnP()
{
    if (selectedSnapIdx < 0 || selectedSnapIdx >= (int)savedSnaps.size() || H_snap_to_cur.empty()) {
        std::cout << "AutoPnP: invalid snapshot/homography.\n"; return;
    }
    const auto& S = savedSnaps[selectedSnapIdx];         // 55 saved pairs for PnP
    if (S.img.size() < 4) { std::cout << "Snapshot has <4 saved points.\n"; return; }

    // 1) Predict where each saved 2D (snapshot pixel) lands in the current frame
    std::vector<cv::Point2f> pred;
    cv::perspectiveTransform(S.img, pred, H_snap_to_cur);   // S.img -> current pixels

    // 2) For each predicted point, snap to the nearest current keypoint (optional, but steadier)
    const float SNAP_RADIUS = 6.0f; // pixels
    lastMatchedObject3D.clear();
    lastMatchedSaved2D.clear();
    lastMatchedCurrent2D.clear();

    for (size_t i=0; i<S.img.size(); ++i) {
        cv::Point2f p = pred[i];
        int best = -1; float bestD2 = SNAP_RADIUS*SNAP_RADIUS;
        for (int j=0; j<(int)curKps.size(); ++j) {
            float dx = curKps[j].pt.x - p.x;
            float dy = curKps[j].pt.y - p.y;
            float d2 = dx*dx + dy*dy;
            if (d2 < bestD2) { bestD2 = d2; best = j; }
        }
        if (best >= 0) {
            // 3D from saved, 2D from current
            const auto& P3 = S.obj[i];
            lastMatchedObject3D.emplace_back(P3.x, P3.y, P3.z);     // glm::vec3
            lastMatchedSaved2D.push_back(S.img[i]);                 // (for prints/lines)
            lastMatchedCurrent2D.push_back(curKps[best].pt);        // snapped 2D
        }
    }

    std::cout << "AutoPnP: matched " << lastMatchedObject3D.size()
              << " / " << S.obj.size() << " saved points.\n";

    // 3) Move them into your PnP containers and compute pose
    coloredPointsLeft.clear();
    image2DPointsRight.clear();
    colorIndexLeft = 0;

    for (size_t i=0; i<lastMatchedObject3D.size(); ++i) {
        glm::vec3 col = colorPaletteLeft[colorIndexLeft % colorPaletteLeft.size()];
        ++colorIndexLeft;
        coloredPointsLeft.push_back({ lastMatchedObject3D[i], col }); // 3D
        image2DPointsRight.push_back( lastMatchedCurrent2D[i] );      // 2D
    }

    if (coloredPointsLeft.size() >= 4) computeCameraPose();
    else std::cout << "AutoPnP: need >= 4 correspondences.\n";
}

bool SaveRightViewportPNG(const std::string& path)
{
    if (!EnsureSnapsDir()) return false;

    cv::Mat frameBGR;
    if (!CaptureRightViewportToBGR(frameBGR)) {
        std::cerr << "SaveRightViewportPNG: capture failed.\n";
        return false;
    }
    if (!cv::imwrite(path, frameBGR)) {
        std::error_code ec;
        std::cerr << "SaveRightViewportPNG: write failed for " << path
                  << " (CWD=" << fs::current_path(ec).string() << ")\n";
        return false;
    }
    std::cout << "💾 Saved snapshot PNG: " << path << "\n";
    return true;
}


bool SaveNextSnapshotPNG()
{
    if (screenshotCounter >= BANK_COUNT) {
        std::cout << "All " << BANK_COUNT << " PNGs already saved.\n";
        return false;
    }
    const std::string path = bankPaths[screenshotCounter];
    if (SaveRightViewportPNG(path)) {
        ++screenshotCounter;               // advance to the next slot
        return true;
    }
    return false;
}

int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Make sure the snapshots dir exists (so cv::imwrite succeeds)
EnsureSnapsDir();

// (Optional: show current working directory for sanity)
std::error_code ec;
std::cout << "CWD = " << fs::current_path(ec).string() << "\n";


// Load saved 2D points from your correspondences file (x,y,z,u,v)

if(map1){
LoadCorrespondencesCSV("correspondences1.csv");
}
else{
LoadCorrespondencesCSV("correspondences2.csv");
}

if (!LoadFeatureBank()) std::cout << "WARNING: feature bank not loaded.\n";




    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Terrain Viewer", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    Shader shader("C:/Users/mohamad/Downloads/BasicOpenGL-main/BasicOpenGL-main/src/7.3.camera.vs", "C:/Users/mohamad/Downloads/BasicOpenGL-main/BasicOpenGL-main/src/7.3.camera.fs");

    if(map2)
    LoadUnstructuredTerrain("res/heightmap2.png", 5000);
    else
    LoadUnstructuredTerrain("res/heightmap1.png", 5000);

    glGenVertexArrays(1, &terrainVAO);
    glGenBuffers(1, &terrainVBO);
    glGenBuffers(1, &terrainEBO);

    glBindVertexArray(terrainVAO);
    glBindBuffer(GL_ARRAY_BUFFER, terrainVBO);
    glBufferData(GL_ARRAY_BUFFER, terrainVertices.size() * sizeof(Vertex), terrainVertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrainEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, terrainIndices.size() * sizeof(unsigned int), terrainIndices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        if (isRecording)
        {
            recordedPositions.push_back(cameraPos);
            recordedFronts.push_back(cameraFront);
        }

        static bool rightArrowLast = false;
        static bool leftArrowLast = false;

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 model = glm::mat4(1.0f);
        if (framePlaybackMode && !savedFrames.empty() && !computed)
        {
            const FrameSnapshot &frame = savedFrames[framePlaybackIndex];
            cameraPos = frame.camPos;
            cameraFront = frame.camFront;
        }

        shader.use();
        shader.setBool("useUniformColor", false);

        // === left model transformation ===
        glm::vec3 globalCamPos;
        if(map2){
            model = glm::translate(model, glm::vec3(0.0f, -50.0f, -200.0f));
            globalCamPos = glm::vec3(400.254, -163.424, 904.804);
        }
        
        else{
            model = glm::translate(model, glm::vec3(0.0f, 200.0f, -300.0f));
            globalCamPos = glm::vec3(600.254, -0.424, 1500.804); // Or tweak it
        }
        model = glm::rotate(model, glm::radians(-65.0f), glm::vec3(1.0f, 0.0f, 0.0f));

        // === LEFT: GLOBAL VIEW ===
        glViewport(0, 0, SCR_WIDTH / 2, SCR_HEIGHT);

        glm::mat4 projectionGlobal = glm::perspective(glm::radians(fov), (float)(SCR_WIDTH / 2) / (float)SCR_HEIGHT, 0.1f, 3000.0f);

        // This camera looks from a fixed position to the center of the scene
        
        glm::mat4 viewGlobal = glm::lookAt(globalCamPos, globalCamPos + glm::vec3(-0.107048, 0.219846, -0.969644), glm::vec3(0.0f, 1.0f, 0.0f));

        shader.setMat4("projection", projectionGlobal);
        shader.setMat4("view", viewGlobal);
        shader.setMat4("model", model);

        shader.use();
        shader.setBool("useUniformColor", false);

        glBindVertexArray(terrainVAO);
        glDrawElements(GL_TRIANGLES, terrainIndices.size(), GL_UNSIGNED_INT, 0);

        // Draw camera path
        shader.use();
        shader.setBool("useUniformColor", true);
        shader.setVec3("color", glm::vec3(1.0f, 1.0f, 1.0f)); // Red
                                                              // Draw arrow/point (VBO only needs positions, color attribute can be dummy/ignored)

        if (!pathLineVertices.empty() && !framePlaybackMode)
        {
            glBindVertexArray(pathVAO);
            glLineWidth(3.0f); // Or 5.0f or whatever you like
            glBindVertexArray(pathVAO);
            glDrawArrays(GL_LINE_STRIP, 0, pathLineVertices.size());
            glLineWidth(3.0f); // Reset after
        }

        glm::vec3 head = cameraPos;
        glm::vec3 dir = -cameraFront; // Camera direction
        glm::vec3 right = glm::normalize(glm::cross(dir, cameraUp));
        glm::vec3 tail = head - dir * 30.0f; // Optional for logic reuse

        float arrowLength = 50.0f; // How far the base is from the tip
        float arrowWidth = 30.0f;  // Width of the arrow base

        glm::vec3 arrowTip = head;
        glm::vec3 base1 = head - dir * arrowLength + right * arrowWidth;
        glm::vec3 base2 = head - dir * arrowLength - right * arrowWidth;

        float arrowVertices[] = {
            arrowTip.x, arrowTip.y, arrowTip.z,
            base1.x, base1.y, base1.z,
            base2.x, base2.y, base2.z};

        unsigned int arrowVAO, arrowVBO;
        glGenVertexArrays(1, &arrowVAO);
        glGenBuffers(1, &arrowVBO);

        glBindVertexArray(arrowVAO);
        glBindBuffer(GL_ARRAY_BUFFER, arrowVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(arrowVertices), arrowVertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0);

        glDrawArrays(GL_TRIANGLES, 0, 3);

        glBindVertexArray(0);
        glDeleteVertexArrays(1, &arrowVAO);
        glDeleteBuffers(1, &arrowVBO);

        for (const auto &arrow : arrows)
        {
            glm::vec3 head = arrow.position;
            glm::vec3 dir = -arrow.direction; // Arrow points in the opposite direction of the camera front
            glm::vec3 right = glm::normalize(glm::cross(dir, arrow.up));
            glm::vec3 tail = head - dir * 30.0f;

            float arrowLength = 50.0f;
            float arrowWidth = 30.0f;

            glm::vec3 arrowTip = head;
            glm::vec3 base1 = head - dir * arrowLength + right * arrowWidth;
            glm::vec3 base2 = head - dir * arrowLength - right * arrowWidth;

            float arrowVertices[] = {
                arrowTip.x, arrowTip.y, arrowTip.z,
                base1.x, base1.y, base1.z,
                base2.x, base2.y, base2.z};

            unsigned int arrowVAO, arrowVBO;
            glGenVertexArrays(1, &arrowVAO);
            glGenBuffers(1, &arrowVBO);

            glBindVertexArray(arrowVAO);
            glBindBuffer(GL_ARRAY_BUFFER, arrowVBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(arrowVertices), arrowVertices, GL_STATIC_DRAW);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
            glEnableVertexAttribArray(0);

            shader.use();
            shader.setBool("useUniformColor", true);
            shader.setVec3("color", glm::vec3(0.0f, 0.0f, 1.0f)); // Blue for arrows
            glDrawArrays(GL_TRIANGLES, 0, 3);

            glBindVertexArray(0);
            glDeleteVertexArrays(1, &arrowVAO);
            glDeleteBuffers(1, &arrowVBO);
        }

        for (const auto &arrow : arrows2)
        {
            glm::vec3 head = arrow.position;
            glm::vec3 dir = -arrow.direction; // Arrow points in the opposite direction of the camera front
            glm::vec3 right = glm::normalize(glm::cross(dir, arrow.up));
            glm::vec3 tail = head - dir * 30.0f;

            float arrowLength = 50.0f;
            float arrowWidth = 30.0f;

            glm::vec3 arrowTip = head;
            glm::vec3 base1 = head - dir * arrowLength + right * arrowWidth;
            glm::vec3 base2 = head - dir * arrowLength - right * arrowWidth;

            float arrowVertices[] = {
                arrowTip.x, arrowTip.y, arrowTip.z,
                base1.x, base1.y, base1.z,
                base2.x, base2.y, base2.z};

            unsigned int arrowVAO, arrowVBO;
            glGenVertexArrays(1, &arrowVAO);
            glGenBuffers(1, &arrowVBO);

            glBindVertexArray(arrowVAO);
            glBindBuffer(GL_ARRAY_BUFFER, arrowVBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(arrowVertices), arrowVertices, GL_STATIC_DRAW);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
            glEnableVertexAttribArray(0);

            shader.use();
            shader.setBool("useUniformColor", true);
            shader.setVec3("color", glm::vec3(0.0f, 1.0f, 0.0f)); // Blue for arrows
            glDrawArrays(GL_TRIANGLES, 0, 3);

            glBindVertexArray(0);
            glDeleteVertexArrays(1, &arrowVAO);
            glDeleteBuffers(1, &arrowVBO);
        }

        // === DRAW CLICKED POINTS IN LEFT VIEW ===
        for (const auto &pt : coloredPointsLeft)
        {
            shader.use();
            shader.setBool("useUniformColor", true);
            shader.setVec3("color", pt.color);

            float ptVertices[] = {pt.pos.x, pt.pos.y, pt.pos.z};
            unsigned int ptVAO, ptVBO;
            glGenVertexArrays(1, &ptVAO);
            glGenBuffers(1, &ptVBO);

            glBindVertexArray(ptVAO);
            glBindBuffer(GL_ARRAY_BUFFER, ptVBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(ptVertices), ptVertices, GL_STATIC_DRAW);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
            glEnableVertexAttribArray(0);

            glPointSize(20.0f);
            glDrawArrays(GL_POINTS, 0, 1);

            glBindVertexArray(0);
            glDeleteVertexArrays(1, &ptVAO);
            glDeleteBuffers(1, &ptVBO);
        }

        // === RIGHT: CAMERA VIEW ===
        glm::mat4 modelC = glm::mat4(1.0f);
        shader.use();
        shader.setBool("useUniformColor", false);

        glViewport(SCR_WIDTH / 2, 0, SCR_WIDTH / 2, SCR_HEIGHT);

        glm::mat4 projectionCam = glm::perspective(glm::radians(fov), (float)(SCR_WIDTH / 2) / (float)SCR_HEIGHT, 0.1f, 3000.0f);
        glm::mat4 viewCam = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

        shader.setMat4("projection", projectionCam);
        shader.setMat4("view", viewCam);
        shader.setMat4("model", modelC);

        // === DRAW CLICKED POINTS IN RIGHT VIEW ===
        for (const auto &pt : coloredPointsRight)
        {
            shader.use();
            shader.setBool("useUniformColor", true);
            shader.setVec3("color", pt.color);

            // Setup VAO/VBO for this point
            float ptVertices[] = {pt.pos.x, pt.pos.y, pt.pos.z};
            unsigned int ptVAO, ptVBO;
            glGenVertexArrays(1, &ptVAO);
            glGenBuffers(1, &ptVBO);

            glBindVertexArray(ptVAO);
            glBindBuffer(GL_ARRAY_BUFFER, ptVBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(ptVertices), ptVertices, GL_STATIC_DRAW);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
            glEnableVertexAttribArray(0);

            glPointSize(20.0f);
            glDrawArrays(GL_POINTS, 0, 1);

            glBindVertexArray(0);
            glDeleteVertexArrays(1, &ptVAO);
            glDeleteBuffers(1, &ptVBO);
        }
        shader.use();
        shader.setBool("useUniformColor", false);

        // IMPORTANT: Re-bind terrain VAO again
        glBindVertexArray(terrainVAO);
        glDrawElements(GL_TRIANGLES, terrainIndices.size(), GL_UNSIGNED_INT, 0);

        // After drawing RIGHT viewport terrain (before swap):
        
        if (requestDetectKeypoints) {
            DetectKeypointsFromRightViewport();   // grabs pixels & runs ORB
            requestDetectKeypoints = false;
        }
        if (showKeypointsOverlay) {
    DrawKeypointOverlay(shader);
}
if (requestMatchK) {
    MatchTopKAndPrint();   // prints pairs + uploads overlay points
    requestMatchK = false;
}
DrawMatchesOverlay(shader); // draws red(old) + cyan(new) top-K

if (requestAutoPnP) {
    // Use the matched sets produced by MatchTopKAndPrint()
    if (lastMatchedObject3D.size() >= 4 &&
        lastMatchedObject3D.size() == lastMatchedCurrent2D.size())
    {
        if (clearPrevPairsOnAuto) {
            coloredPointsLeft.clear();
            image2DPointsRight.clear();
            colorIndexLeft = 0;  // optional: reset coloring
        }

        // Fill LEFT (3D) and RIGHT (2D pixels) used by computeCameraPose()
        for (size_t i = 0; i < lastMatchedObject3D.size(); ++i) {
            const glm::vec3& P = lastMatchedObject3D[i];  // world-space 3D
            const cv::Point2f& uv = lastMatchedSaved2D[i]; // RIGHT pixels

            glm::vec3 col = colorPaletteLeft[colorIndexLeft % colorPaletteLeft.size()];
            ++colorIndexLeft;
            coloredPointsLeft.push_back({P, col});  // 3D points for PnP
            image2DPointsRight.push_back(uv);       // 2D pixels for PnP
        }

        // IMPORTANT: do NOT push pixel coords into coloredPointsRight (that list is world-space)
        computeCameraPose();
    } else {
        std::cout << "AutoPnP: not enough matched pairs (" << lastMatchedObject3D.size()
                  << "). Need >= 4.\n";
    }
    requestAutoPnP = false;
}


              // overlays green dots in RIGHT viewport


        glfwSwapBuffers(window);
        glfwPollEvents();
        //kpOverlay.clear();
    }

    glDeleteVertexArrays(1, &terrainVAO);
    glDeleteBuffers(1, &terrainVBO);
    glDeleteBuffers(1, &terrainEBO);

    glfwTerminate();
    return 0;
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow *window)
{
    float cameraSpeed = 300.5f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;

          
        static bool kPressedLast = false;
if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS && !kPressedLast) {
    kPressedLast = true;

            if (isRecording)
        {
            Arrow newArrow;
            newArrow.position = cameraPos;
            newArrow.direction = cameraFront;
            newArrow.up = cameraUp;
            arrows.push_back(newArrow);
            std::cout << "📸 Arrow created! Total arrows: " << arrows.size() << "\n";

            FrameSnapshot snapshot;
            snapshot.camPos = cameraPos;
            snapshot.camFront = cameraFront;
            snapshot.modelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -50.0f, -200.0f));
            snapshot.modelMatrix = glm::rotate(snapshot.modelMatrix, glm::radians(-65.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            savedFrames.push_back(snapshot);
            std::cout << "📸 Snapshot captured! Total: " << savedFrames.size() << "\n";
        }
    requestDetectKeypoints = true;   // your render loop will call DetectKeypointsFromRightViewport() 
    // 1) Detect ~2500 ORB features in the current right viewport
    DetectKeypointsFromRightViewport();

    // 2) If the bank isn’t loaded yet (first time), try to load now
    if (featBank.empty() || featBank.size() != BANK_COUNT || featBank[0].desc.empty()) {
        if (!LoadFeatureBank()) {
            std::cout << "K: feature bank not ready. Save 9 PNGs (press B), then press L.\n";
            goto K_DONE;
        }
    }

    // 3) Choose closest snapshot via descriptor matching + RANSAC
    if (SelectClosestSnapshotByFeatures() >= 0) {
        // 4) Build 3D-2D pairs from that snapshot’s 55 saved points and run PnP
        BuildPairsFromSelectedSnapshotAndRunPnP();
    } else {
        std::cout << "K: could not select a snapshot.\n";
    }

K_DONE: ;
}
if (glfwGetKey(window, GLFW_KEY_K) == GLFW_RELEASE) kPressedLast = false;





    static bool tPressedLastFrame = false;
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS && !tPressedLastFrame)
    {
        tPressedLastFrame = true;
        useComputedPose = !useComputedPose; // Toggle between original and computed poses
        if (useComputedPose)
        {
            cameraPos = computedPose.position;
            cameraFront = computedPose.front;
            cameraUp = computedPose.up;
            std::cout << "Switched to computed camera pose.\n";
            Arrow newArrow;
            newArrow.position = cameraPos;
            newArrow.direction = cameraFront;
            newArrow.up = cameraUp;
            arrows2.push_back(newArrow);
            std::cout << "📸 Arrow created! Total arrows: " << arrows2.size() << "\n";
        }
        else
        {
            cameraPos = originalPose.position;
            cameraFront = originalPose.front;
            cameraUp = originalPose.up;
            std::cout << "Switched to original camera pose.\n";
        }
    }
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_RELEASE)
    {
        tPressedLastFrame = false;
    }

    static bool cPressedLastFrame = false;
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS && !cPressedLastFrame)
    {
        cPressedLastFrame = true;
        computed = !computed; // Toggle computed camera pose
        computeCameraPose();
    }
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_RELEASE)
    {
        cPressedLastFrame = false;
    }
    static bool rPressedLastFrame = false;

    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS && !rPressedLastFrame)
    {
        rPressedLastFrame = true;
        isRecording = !isRecording;

        if (isRecording)
        {
            recordedPositions.clear();
            recordedFronts.clear();
            std::cout << "📹 Started Recording\n";
        }
        else
        {
            std::cout << "✅ Stopped Recording\n";
        }
    }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_RELEASE)
        rPressedLastFrame = false;

    if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS)
    {
        mouseArrow = !mouseArrow;
        if (mouseArrow)
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        else
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); // or HIDDEN/DISABLED
    }

    // Build path line from recorded positions
    pathLineVertices = recordedPositions;

    if (pathVAO == 0)
    {
        glGenVertexArrays(1, &pathVAO);
        glGenBuffers(1, &pathVBO);
    }

    glBindVertexArray(pathVAO);
    glBindBuffer(GL_ARRAY_BUFFER, pathVBO);
    glBufferData(GL_ARRAY_BUFFER, pathLineVertices.size() * sizeof(glm::vec3), pathLineVertices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);
    glEnableVertexAttribArray(0);

    static bool pPressedLastFrame = false;

    static bool bPressedLast = false;
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS && !bPressedLast)
    {
        bPressedLast = true;

        if (isRecording)
        {
            Arrow newArrow;
            newArrow.position = cameraPos;
            newArrow.direction = cameraFront;
            newArrow.up = cameraUp;
            arrows.push_back(newArrow);
            std::cout << "📸 Arrow created! Total arrows: " << arrows.size() << "\n";

            FrameSnapshot snapshot;
            snapshot.camPos = cameraPos;
            snapshot.camFront = cameraFront;
            snapshot.modelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -50.0f, -200.0f));
            snapshot.modelMatrix = glm::rotate(snapshot.modelMatrix, glm::radians(-65.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            savedFrames.push_back(snapshot);
            std::cout << "📸 Snapshot captured! Total: " << savedFrames.size() << "\n";
        }

            // Toggle feature overlay on/off
            showKeypointsOverlay = !showKeypointsOverlay;
            if (showKeypointsOverlay) {
                // only detect when turning ON
                requestDetectKeypoints = true;
            } else {
                // turning OFF: clear buffers so nothing is drawn
                kpOverlay.clear();
                if (kpVAO) { glDeleteVertexArrays(1, &kpVAO); kpVAO = 0; }
                if (kpVBO) { glDeleteBuffers(1, &kpVBO); kpVBO = 0; }
            }
                // Save right viewport PNG into res/snaps/snap_#.png (up to 9)
            SaveNextSnapshotPNG();



    }
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_RELEASE)
        bPressedLast = false;

    static bool shiftRPressed = false;
    if ((glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) &&
        glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS && !shiftRPressed)
    {

        shiftRPressed = true;
        if (framePlaybackMode)
        {
            framePlaybackMode = false;
            std::cout << "⏸️ Snapshot Playback Stopped\n";
        }
        else if (!savedFrames.empty())
        {
            framePlaybackMode = true;
            framePlaybackIndex = 0;
            std::cout << "🖼️ Snapshot Playback Started\n";
        }
    }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_RELEASE)
        shiftRPressed = false;

    if (framePlaybackMode)
    {
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS && framePlaybackIndex < savedFrames.size() - 1)
        {
            framePlaybackIndex++;
            std::cout << "➡️ Frame: " << framePlaybackIndex << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS && framePlaybackIndex > 0)
        {
            framePlaybackIndex--;
            std::cout << "⬅️ Frame: " << framePlaybackIndex << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    static bool pKeyPressed = false;
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS && !pKeyPressed && framePlaybackMode)
    {
        if (computed)
            computed = false; // Stop computed camera pose if picking is toggled;
        pKeyPressed = true;
        PickingMode = !PickingMode;

        if (PickingMode)
        {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            // pickingState = 1;  // Start picking: wait for global view click
            std::cout << "🎯 Picking started. Click on the global view.\n";
        }
        else
        {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            std::cout << "❌ Picking stopped.\n";
        }
    }
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_RELEASE)
    {
        pKeyPressed = false;
    }

    bool pressedX = false;
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS && !pressedX)
    {
        pressedX = true;
        std::cout << "Camera Position: (" << cameraPos.x << ", " << cameraPos.y << ", " << cameraPos.z << ")\n";
    }
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_RELEASE)
    {
        pressedX = true;
    }
}

void mouse_callback(GLFWwindow *window, double xposIn, double yposIn)
{
    if (!computed)
    {
        float xpos = static_cast<float>(xposIn);
        float ypos = static_cast<float>(yposIn);

        if (firstMouse)
        {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }

        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos;
        lastX = xpos;
        lastY = ypos;

        float sensitivity = 0.1f;
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        yaw += xoffset;
        pitch += yoffset;

        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;

        glm::vec3 front;
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraFront = glm::normalize(front);
    }
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    fov -= (float)yoffset;
}

// === MODIFIED: Mouse button callback to add point in global view ===
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && PickingMode)
    {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        bool isLeftView = xpos < SCR_WIDTH / 2;

        if (isLeftView)
        {
            // LEFT VIEW: Add colored point for global view
            glm::mat4 modelMat = glm::mat4(1.0f);
            glm::vec3 ray_origin;
            if(map2){
                modelMat = glm::translate(modelMat, glm::vec3(0.0f, -50.0f, -200.0f));
                ray_origin = glm::vec3(400.254, -163.424, 904.804);
            }
            else{
                modelMat = glm::translate(modelMat, glm::vec3(0.0f, 200.0f, -300.0f));
                ray_origin = glm::vec3(600.254, -0.424, 1500.804);
            }
            
            modelMat = glm::rotate(modelMat, glm::radians(-65.0f), glm::vec3(1.0f, 0.0f, 0.0f));

            
            glm::mat4 projection = glm::perspective(glm::radians(fov), (float)(SCR_WIDTH / 2) / (float)SCR_HEIGHT, 0.1f, 3000.0f);
            glm::mat4 view = glm::lookAt(ray_origin, ray_origin + glm::vec3(-0.107048, 0.219846, -0.969644), glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 invModel = glm::inverse(modelMat);

            float x_ndc = (2.0f * xpos) / (SCR_WIDTH / 2) - 1.0f;
            float y_ndc = 1.0f - (2.0f * ypos) / SCR_HEIGHT;

            glm::vec4 ray_clip = glm::vec4(x_ndc, y_ndc, -1.0f, 1.0f);
            glm::vec4 ray_eye = glm::inverse(projection) * ray_clip;
            ray_eye.z = -1.0f;
            ray_eye.w = 0.0f;

            glm::vec3 ray_world = glm::normalize(glm::vec3(glm::inverse(view) * ray_eye));
            

            glm::vec3 ray_origin_model = glm::vec3(invModel * glm::vec4(ray_origin, 1.0f));
            glm::vec3 ray_dir_model = glm::normalize(glm::vec3(invModel * glm::vec4(ray_world, 0.0f)));

            glm::vec3 intersection;
            bool hit = false;
            for (size_t i = 0; i < terrainIndices.size(); i += 3)
            {
                glm::vec3 v0 = glm::vec3(terrainVertices[terrainIndices[i + 0]].x,
                                         terrainVertices[terrainIndices[i + 0]].y,
                                         terrainVertices[terrainIndices[i + 0]].z);
                glm::vec3 v1 = glm::vec3(terrainVertices[terrainIndices[i + 1]].x,
                                         terrainVertices[terrainIndices[i + 1]].y,
                                         terrainVertices[terrainIndices[i + 1]].z);
                glm::vec3 v2 = glm::vec3(terrainVertices[terrainIndices[i + 2]].x,
                                         terrainVertices[terrainIndices[i + 2]].y,
                                         terrainVertices[terrainIndices[i + 2]].z);

                float t;
                glm::vec3 hitPoint;
                if (RayIntersectsTriangle(ray_origin_model, ray_dir_model, v0, v1, v2, t, hitPoint))
                {
                    intersection = hitPoint;
                    hit = true;
                    break;
                }
            }
            if (hit)
            {
                glm::vec3 color = colorPaletteLeft[colorIndexLeft % colorPaletteLeft.size()];
                colorIndexLeft++;
                coloredPointsLeft.push_back({intersection, color}); // Add to left points
                std::cout << "[LEFT] Point added at: (" << intersection.x << ", " << intersection.y << ", " << intersection.z << ")\n";
                pendingLeft = intersection;
                hasPendingLeft = true;
                if (hasPendingRight) {
                    appendCorrespondence(pendingLeft, pendingRight);
                    hasPendingLeft = hasPendingRight = false;
                } else {
                    std::cout << "…waiting for RIGHT click to pair & save.\n";
                }

            }
            else
            {
                std::cout << "[LEFT] No intersection with terrain.\n";
            }
        }
        else
        {
            // RIGHT VIEW: Add colored point for camera view
            float x_ndc = (2.0f * (xpos - SCR_WIDTH / 2)) / (SCR_WIDTH / 2) - 1.0f;
            float y_ndc = 1.0f - (2.0f * ypos) / SCR_HEIGHT;

            glm::vec4 ray_clip = glm::vec4(x_ndc, y_ndc, -1.0f, 1.0f);
            glm::mat4 projection = glm::perspective(glm::radians(fov), (float)(SCR_WIDTH / 2) / (float)SCR_HEIGHT, 0.1f, 3000.0f);
            glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

            glm::vec4 ray_eye = glm::inverse(projection) * ray_clip;
            ray_eye.z = -1.0f;
            ray_eye.w = 0.0f;

            glm::vec3 ray_world = glm::normalize(glm::vec3(glm::inverse(view) * ray_eye));
            glm::vec3 ray_origin = cameraPos;

            glm::vec3 intersection;
            bool hit = false;
            for (size_t i = 0; i < terrainIndices.size(); i += 3)
            {
                glm::vec3 v0 = glm::vec3(terrainVertices[terrainIndices[i + 0]].x,
                                         terrainVertices[terrainIndices[i + 0]].y,
                                         terrainVertices[terrainIndices[i + 0]].z);
                glm::vec3 v1 = glm::vec3(terrainVertices[terrainIndices[i + 1]].x,
                                         terrainVertices[terrainIndices[i + 1]].y,
                                         terrainVertices[terrainIndices[i + 1]].z);
                glm::vec3 v2 = glm::vec3(terrainVertices[terrainIndices[i + 2]].x,
                                         terrainVertices[terrainIndices[i + 2]].y,
                                         terrainVertices[terrainIndices[i + 2]].z);

                float t;
                glm::vec3 hitPoint;
                if (RayIntersectsTriangle(ray_origin, ray_world, v0, v1, v2, t, hitPoint))
                {
                    intersection = hitPoint;
                    hit = true;
                    break;
                }
            }
            if (hit)
                {
                    // color for both views (keep them visually linked)
                    glm::vec3 color = colorPaletteRight[colorIndexRight % colorPaletteRight.size()];
                    colorIndexRight++;

                    // 1) RIGHT view viz (as before)
                    coloredPointsRight.push_back({intersection, color});

                    // 2) RIGHT pixel coordinate (for PnP)
                    cv::Point2f imgPt((float)(xpos - SCR_WIDTH / 2.0f), (float)ypos);
                    image2DPointsRight.push_back(imgPt);

                    // 3) AUTO-PAIR LEFT: mirror the same 3D hit to the left list
                    if (autoPairLeftFromRight) {
                        glm::vec3 left3D = intersection; // default to geometry
                        // Try saved snapshot mapping first; fall back to geometry if not found/too far
                        float dpx = 0.f;
                        if (pickSaved3DFromSnapshot(imgPt.x, imgPt.y, left3D, &dpx) && dpx < 25.0f) {
                            // good snap match within 25 px
                        }
                        coloredPointsLeft.push_back({left3D, color});
                        appendCorrespondence(left3D, imgPt);
                        counter ++;
                        std::cout << "counter: " << counter << "\n";
                    }


                    std::cout << "[RIGHT] Hit at 3D (" << intersection.x << ", " << intersection.y << ", " << intersection.z
                            << "), pixel (" << imgPt.x << ", " << imgPt.y << ")\n";
                }
                

            else
            {
                std::cout << "[RIGHT] No intersection with terrain.\n";
            }
        }
    }
}

bool RayIntersectsTriangle(
    const glm::vec3 &orig, const glm::vec3 &dir,
    const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2,
    float &t, glm::vec3 &hitPoint)
{
    const float EPSILON = 0.0000000001f;
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 h = glm::cross(dir, edge2);
    float a = glm::dot(edge1, h);
    if (fabs(a) < EPSILON)
        return false;

    float f = 1.0f / a;
    glm::vec3 s = orig - v0;
    float u = f * glm::dot(s, h);
    if (u < 0.0f || u > 1.0f)
        return false;

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(dir, q);
    if (v < 0.0f || u + v > 1.0f)
        return false;

    t = f * glm::dot(edge2, q);
    if (t > EPSILON)
    {
        hitPoint = orig + dir * t;
        return true;
    }

    return false;
}
