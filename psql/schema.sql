CREATE SCHEMA IF NOT EXISTS YT;

-- Drop existing tables (in reverse dependency order)
DROP TABLE IF EXISTS YT.Comments CASCADE;
DROP TABLE IF EXISTS YT.Videos CASCADE;
DROP TABLE IF EXISTS YT.Users CASCADE;
DROP TABLE IF EXISTS YT.Channels CASCADE;
DROP TABLE IF EXISTS YT.Versions CASCADE;

CREATE TABLE YT.Versions (
    -- For data lineage/versioning
    versionName VARCHAR(24) PRIMARY KEY,
    createdAt TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    versionDescription TEXT
);

CREATE TABLE YT.Channels (
    channelId VARCHAR(30) PRIMARY KEY
);

CREATE TABLE YT.Users (
    userId VARCHAR(20) PRIMARY KEY,
    username VARCHAR(40) NOT NULL,
    createDate TIMESTAMP NOT NULL,
    subCount INTEGER,
    videoCount INTEGER
);

CREATE TABLE YT.Videos (
    videoId VARCHAR(20) PRIMARY KEY,
    title TEXT NOT NULL,
    publishDate TIMESTAMP NOT NULL,
    channelId VARCHAR(30) NOT NULL,
    versionName VARCHAR(24) NOT NULL,

    FOREIGN KEY (channelId) REFERENCES YT.Channels(channelId),
    FOREIGN KEY (versionName) REFERENCES YT.Versions(versionName)
);

CREATE TABLE YT.Comments (
    commentId VARChAR(50) PRIMARY KEY,
    commenterId VARCHAR(20) NOT NULL,
    videoId VARCHAR(20) NOT NULL,
    isReply BOOLEAN NOT NULL,
    threadId VARCHAR(50),
    publishDate TIMEStAMP NOT NULL,
    editDate TIMESTAMP,
    likeCount INTEGER,
    commentText TEXT NOT NULL,
    versionName VARCHAR(24) NOT NULL,

    FOREIGN KEY (commenterId) REFERENCES YT.Users(userId)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    FOREIGN KEY (videoId) REFERENCES YT.Videos(videoId)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    FOREIGN KEY (threadId) REFERENCES YT.Comments(commentId)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    FOREIGN KEY (versionName) REFERENCES YT.Versions(versionName)
);

CREATE INDEX idx_comments_commenter ON YT.Comments(commenterId);
CREATE INDEX idx_comments_video ON YT.Comments(videoId);
CREATE INDEX idx_comments_thread ON YT.Comments(threadId);